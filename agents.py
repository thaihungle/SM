import numpy as np
from collections import deque
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim

# from torch.distributions.categorical import Categorical
from dist import Categorical, DiagGaussian, Bernoulli

from model import CnnActorCriticNetwork, RNDModel, RND_SM_Model, M_Memory
from utils import global_grad_norm_


class RNDAgent(object):
    def __init__(
            self,
            input_size,
            action_space,
            num_env,
            num_step,
            gamma,
            lam=0.95,
            learning_rate=1e-4,
            ent_coef=0.01,
            clip_grad_norm=0.5,
            epoch=3,
            batch_size=128,
            ppo_eps=0.1,
            update_proportion=0.25,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False,
            args =None
            ):

        self.num_env = num_env
        if args['UseScaler']:
            input_size = list(input_size)
            input_size[0]+=1
        print(input_size)

        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.epoch = epoch
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.ent_coef = ent_coef
        self.ppo_eps = ppo_eps
        self.clip_grad_norm = clip_grad_norm
        self.update_proportion = update_proportion
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.policy_losses = deque(maxlen=200)
        self.reward_losses = deque(maxlen=200)
      

        self.args = args
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(num_outputs, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(num_outputs, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(num_outputs, num_outputs)
        else:
            raise NotImplementedError
        print("NUM OUTPUT", num_outputs)
        self.output_size = num_outputs
        self.model = CnnActorCriticNetwork(input_size, self.output_size, use_noisy_net)
        self.reward_model = RNDModel(input_size, self.output_size)
        self.reward_model = self.reward_model.to(self.device)
        self.model = self.model.to(self.device)
        self.dist = self.dist.to(self.device)
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.reward_model.parameters())+list(self.dist.parameters()),
                                    lr=learning_rate)

    def get_action(self, state, deterministic=False):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        policy, value_ext, value_int = self.model(state)
        dist = self.dist(policy)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return action.data.cpu().numpy(), value_ext.data.cpu().numpy().squeeze(), value_int.data.cpu().numpy().squeeze(), policy.detach()

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def compute_intrinsic_reward(self, next_obs):
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        target_next_feature = self.reward_model.target(next_obs)
        predict_next_feature = self.reward_model.predictor(next_obs)
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2
        return intrinsic_reward.data.cpu().numpy()

    def train_model(self, s_batch, target_ext_batch, target_int_batch, y_batch, adv_batch, next_obs_batch, old_policy, args):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        target_ext_batch = torch.FloatTensor(target_ext_batch).to(self.device)
        target_int_batch = torch.FloatTensor(target_int_batch).to(self.device)
        y_batch = torch.FloatTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)

        sample_range = np.arange(len(s_batch))
        forward_mse = nn.MSELoss(reduction='none')

        with torch.no_grad():
            policy_old_list = torch.stack(old_policy).permute(1, 0, 2).contiguous().view(-1, self.output_size).to(
                self.device)

            m_old = self.dist(policy_old_list)
            log_prob_old = m_old.log_prob(y_batch).detach()
            # ------------------------------------------------------------

        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]
                # --------------------------------------------------------------------------------
                # for Curiosity-driven(Random Network Distillation)
                if args['IntCoef']<=0:
                    forward_loss = 0
                else:
                    predict_next_state_feature, target_next_state_feature = self.reward_model(
                        next_obs_batch[sample_idx])
                    forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)

                    # Proportion of exp used for predictor update
                    mask = torch.rand(len(forward_loss)).to(self.device)
                    mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                    forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
                # ---------------------------------------------------------------------------------

                policy, value_ext, value_int = self.model(s_batch[sample_idx])
                m = self.dist(policy)
                log_prob = m.log_prob(y_batch[sample_idx])
                ratio = torch.exp(log_prob - log_prob_old[sample_idx]).squeeze(-1)

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_ext_loss = F.mse_loss(value_ext.sum(1), target_ext_batch[sample_idx])
                critic_int_loss = F.mse_loss(value_int.sum(1), target_int_batch[sample_idx])

                critic_loss = critic_ext_loss + critic_int_loss

                entropy = m.entropy().mean()
                self.policy_losses.append(actor_loss.item())
                if forward_loss is not 0:
                    self.reward_losses.append(forward_loss.item())
                else:
                    self.reward_losses.append(0)
                self.optimizer.zero_grad()
                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy + forward_loss
                loss.backward()
                global_grad_norm_(list(self.model.parameters()) + list(self.reward_model.predictor.parameters()))
                self.optimizer.step()


class RND_SMAgent(object):
    def __init__(
            self,
            input_size,
            action_space,
            num_env,
            num_step,
            gamma,
            lam=0.95,
            learning_rate=1e-4,
            ent_coef=0.01,
            clip_grad_norm=0.5,
            epoch=3,
            batch_size=128,
            ppo_eps=0.1,
            update_proportion=0.25,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False,
            feature_size = 512,
            item_size =128,
            args=None):
        self.num_env = num_env
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.epoch = epoch
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.ent_coef = ent_coef
        self.ppo_eps = ppo_eps
        self.clip_grad_norm = clip_grad_norm
        self.update_proportion = update_proportion
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.feature_size = feature_size
        self.policy_losses = deque(maxlen=200)
        self.reward_losses = deque(maxlen=200)
        self.slow_rewards = deque(maxlen=200)
        self.fast_rewards = deque(maxlen=200)
        self.all_losses = deque(maxlen=200)


        self.global_step = 0
        self.args = args

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(num_outputs, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(num_outputs, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(num_outputs, num_outputs)
        else:
            raise NotImplementedError

        self.output_size = num_outputs

        if "noM" not in self.args['MemType']:
            self.M_memory = M_Memory(num_env, self.args['MemLength'], feature_size, item_size, self.device, total_time = args['MaxEnvStep'])
            self.s2i=nn.Linear(512, item_size).to(self.device)
            self.reward_model = RND_SM_Model(input_size, self.output_size, item_size)
        else:
            self.reward_model = RND_SM_Model(input_size, self.output_size, 0)

        self.model = CnnActorCriticNetwork(input_size, self.output_size, use_noisy_net)

        self.reward_model = self.reward_model.to(self.device)
        self.dist = self.dist.to(self.device)
        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(list(self.model.parameters())+list(self.reward_model.parameters())+list(self.dist.parameters()),
                                    lr=learning_rate)
  
        self.log_state = []
        self.log_surp = []

    def get_action(self, state, deterministic=False):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        policy, value_ext, value_int = self.model(state)

        dist = self.dist(policy)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()


        return action.data.cpu().numpy(), value_ext.data.cpu().numpy().squeeze(), value_int.data.cpu().numpy().squeeze(), policy.detach()

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)


    def get_feature(self, state):
        feature = self.reward_model.get_feature(state)
        return feature

    def get_feature_fix(self, state):
        feature = self.reward_model.get_feature_fix(state)
        return feature

    def update_M_memory(self, state, done):
        state = torch.FloatTensor(state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        if "noM" not in self.args['MemType']:
            self.M_memory.add_item(self.s2i(state), done)
        return


    def compute_intrinsic_reward(self, next_obs):
        '''
        next_obs: BxS
        '''
        feature = self.get_feature(torch.FloatTensor(next_obs).to(self.device))
       
        target = self.get_feature_fix(torch.FloatTensor(next_obs).to(self.device))
    
      
        pfeature = self.reward_model.predict_feature(torch.nn.functional.dropout(feature, p=self.args['MemNoise']))
        surprise_vector = pfeature - target
        rnd_reward = torch.sum(surprise_vector.pow(2), dim=-1, keepdim=True)

        self.log_state.append(feature[0].detach().cpu().numpy())
        self.log_surp.append(surprise_vector[0].detach().cpu().numpy())

        if "noM" not in self.args['MemType']:
            M_reward, recall, query = self.M_memory.retrieve(self.s2i(surprise_vector), noise=self.args["MemNoise"])
            u_e = recall+query
            surprise_novelty, _ = self.reward_model.get_surprise_novelty(torch.cat([surprise_vector, u_e], dim=-1))
        else:
            surprise_novelty, _ = self.reward_model.get_surprise_novelty(surprise_vector)

        W_reward = torch.sum(surprise_novelty.pow(2), dim=-1, keepdim=True)

        if "noW" in self.args['MemType'] and "noM" in self.args['MemType']:
            intrinsic_reward = rnd_reward
        elif "noW" in self.args['MemType']:
            intrinsic_reward = M_reward #Bx1
        else:
            intrinsic_reward = W_reward #Bx1
       
        self.slow_rewards.append(torch.mean(W_reward).item())
        if "noM" in self.args['MemType']:
            M_reward = W_reward
        self.fast_rewards.append(torch.mean(M_reward).item())

        return intrinsic_reward.data.cpu().numpy(), M_reward.data.cpu().numpy(), surprise_vector.detach(), torch.sum(surprise_vector.pow(2), dim=-1, keepdim=True).detach().cpu().numpy()

    def train_model(self, s_batch, target_ext_batch, target_int_batch, y_batch, adv_batch, next_obs_batch, old_policy, args):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        target_ext_batch = torch.FloatTensor(target_ext_batch).to(self.device)
        target_int_batch = torch.FloatTensor(target_int_batch).to(self.device)
        y_batch = torch.FloatTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)

      
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)

        sample_range = np.arange(len(s_batch))
        forward_mse = nn.MSELoss(reduction='none')
        ce = nn.CrossEntropyLoss()

        if "noM" not in self.args['MemType']:
            self.M_memory.collect_memory()
        with torch.no_grad():
            policy_old_list = torch.stack(old_policy).permute(1, 0, 2).contiguous().view(-1, self.output_size).to(
                self.device)
            m_old = self.dist(policy_old_list)
            log_prob_old = m_old.log_prob(y_batch).detach()
            # ------------------------------------------------------------

        for i in range(self.epoch):

            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                # --------------------------------------------------------------------------------
                # for Curiosity-driven
                indices = torch.LongTensor(sample_idx).to(device=self.device)
                input_state = next_obs_batch[sample_idx]
                forward_loss = 0
                if args['IntCoef']>0:
                    feature = self.get_feature(input_state) #B*Nxd


                    target = self.get_feature_fix(input_state).to(self.device)
                    pfeature = self.reward_model.predict_feature(torch.nn.functional.dropout(feature, p=self.args['MemNoise']))
                    SG_loss = forward_mse(pfeature, target.detach()) #B*Nx1
                    forward_loss = SG_loss.mean(-1)
                    self.reward_losses.append(forward_loss.sum().item())
                    surprise_vector = pfeature - target
                    if "noM" not in self.args['MemType']:
                        true_recall, L_M, query = self.M_memory.retrieve_train(self.s2i(surprise_vector), indices, noise=self.args["MemNoise"])
                        u_e = L_M + query
                        forward_loss += self.M_memory.coef*forward_mse(true_recall, self.s2i(surprise_vector.detach())).mean(-1)
                        _, L_W = self.reward_model.get_surprise_novelty(torch.cat([surprise_vector, u_e], dim=-1))
                    else:
                        _, L_W = self.reward_model.get_surprise_novelty(surprise_vector)
                    
                    
                    if "noW" not in self.args['MemType']:
                        forward_loss += forward_mse(L_W[0], L_W[1].detach()).mean(-1)

                    mask = torch.rand(len(forward_loss)).to(self.device)
                    mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                    forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
                else:
                    forward_loss = 0


                if forward_loss is not 0:
                    self.all_losses.append(forward_loss.item())
                else:
                    self.all_losses.append(0)

                # ---------------------
                #------------------------------------------------------------
                # for PPO
                policy, value_ext, value_int = self.model(s_batch[sample_idx])
             

                m = self.dist(policy)
                log_prob = m.log_prob(y_batch[sample_idx])
                ratio = torch.exp(log_prob - log_prob_old[sample_idx]).squeeze(-1)

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_ext_loss = F.mse_loss(value_ext.sum(1), target_ext_batch[sample_idx])
                critic_int_loss = F.mse_loss(value_int.sum(1), target_int_batch[sample_idx])

                critic_loss = critic_ext_loss + critic_int_loss

                entropy = m.entropy().mean()

                self.optimizer.zero_grad()
                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy + forward_loss
                self.policy_losses.append(actor_loss.item())


                loss.backward()
                self.optimizer.step()
