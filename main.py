from agents import *
from envs import *
from utils import *
from torch.multiprocessing import Pipe

from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import numpy as np
from datetime import datetime
import socket, os, random
import argparse
from argparse import ArgumentParser
from tqdm import tqdm
from collections import deque
import json
import pickle
import seaborn as sns






def main(args):
    # print({section: dict(args[section]) for section in args})
    args = vars(args)
    print(args)
    train_method = args['TrainMethod']
    env_id = args['EnvID']
    torch.manual_seed(args['Seed'])
    random.seed(args['Seed'])
    np.random.seed(args['Seed'])

    if  args['EnvType'] == 'atari':
        env = gym.make(env_id)
    elif 'gymvec' in args['EnvType']:
        env = gym.make(env_id)
        if 'minigrid' in env_id.lower():
            env = FlatObsWrapper(env)
    else:
        raise NotImplementedError

    print(env.observation_space)
    if env.observation_space.shape:
        input_size = env.observation_space.shape  # 4
    else:
        input_size = env.observation_space['observation'].shape # 4

    print(env.action_space)
   
    env.close()


    is_load_model = False
    is_render = False

    import os, shutil
    model_folder = '{}/{}-{}'.format(args['savedir'], env_id, args['TrainMethod']+args['MemType'])
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)

    model_path = '{}/model.pt'.format(model_folder)
    dist_path = '{}/dist.pt'.format(model_folder)
    fast_reward_model_path = '{}/frewardmodel.pt'.format(model_folder)
    target_path = '{}/target.pt'.format(model_folder)
    reward_model_path = '{}/rewardmodel.pt'.format(model_folder)
    reward_norm_path = '{}/rnorm.pt'.format(model_folder)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
                args['logdir'],env_id,
        f"{args['TrainMethod']}{args['IntCoef']}{args['MemType']}{current_time}_{socket.gethostname()}_S{args['Seed']}")





    param_dir = logdir+"/param.txt"

    use_cuda = args['UseGPU']
    use_gae = args['UseGAE']
    use_noisy_net = args['UseNoisyNet']

    lam = float(args['Lambda'])
    num_worker = int(args['NumEnv'])
    save_image_dir = None
    if args['Mode'] == 'test':
        num_worker=1
        save_image_dir = '{}/{}/'.format(model_folder, 'state_imgs')
    num_step = int(args['NumStep'])

    ppo_eps = float(args['PPOEps'])
    epoch = int(args['Epoch'])
    mini_batch = int(args['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(args['LearningRate'])
    entropy_coef = float(args['Entropy'])
    gamma = float(args['Gamma'])
    int_gamma = float(args['IntGamma'])
    clip_grad_norm = float(args['ClipGradNorm'])
    ext_coef = float(args['ExtCoef'])
    int_coef = float(args['IntCoef'])

    sticky_action = args['StickyAction']
    action_prob = float(args['ActionProb'])
    life_done = args['LifeDone']

    input_size_dim = input_size[0]
    if args['UseScaler']:
        input_size_dim+=1

    reward_rms = RunningMeanStd()
    if 'vec' in args['EnvType']:
        obs_rms = RunningMeanStd(shape=(1, input_size_dim))
    else:
        obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    pre_obs_norm_step = int(args['ObsNormStep'])
    discounted_reward = RewardForwardFilter(int_gamma)


    if args['EnvType'] == 'atari':
        env_type = AtariEnvironment
    elif args['EnvType'] == 'gymvec':
        env_type = GymVectorEnvironment
    elif args['EnvType'] == 'gymveccont':
        env_type = GymVectorContEnvironment
    else:
        raise NotImplementedError

    if args['TrainMethod'] == 'RND':
        agent = RNDAgent(
            input_size,
            env.action_space,
            num_worker,
            num_step,
            gamma,
            lam=lam,
            learning_rate=learning_rate,
            ent_coef=entropy_coef,
            clip_grad_norm=clip_grad_norm,
            epoch=epoch,
            batch_size=batch_size,
            ppo_eps=ppo_eps,
            use_cuda=use_cuda,
            use_gae=use_gae,
            use_noisy_net=use_noisy_net,
            args=args
        )
    if args['TrainMethod'] == 'RND_SM':
        agent = RND_SMAgent(
            input_size,
            env.action_space,
            num_worker,
            num_step,
            gamma,
            lam=lam,
            learning_rate=learning_rate,
            ent_coef=entropy_coef,
            clip_grad_norm=clip_grad_norm,
            epoch=epoch,
            batch_size=batch_size,
            ppo_eps=ppo_eps,
            use_cuda=use_cuda,
            use_gae=use_gae,
            use_noisy_net=use_noisy_net,
            feature_size=args['FeatureSize'],
            item_size=args['MemSize'],
            args = args
        )

    if is_load_model or args['Mode']=='test':
        print('load model...')
        if use_cuda:
            agent.model.load_state_dict(torch.load(model_path))
            # agent.dist.load_state_dict(torch.load(dist_path))
            agent.reward_model.load_state_dict(torch.load(reward_model_path))
            # agent.fast_reward_model.load_state_dict(torch.load(fast_reward_model_path))
        else:
            agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            agent.dist.load_state_dict(torch.load(dist_path, map_location='cpu'))
            agent.reward_model.load_state_dict(torch.load(reward_model_path, map_location='cpu'))
            # agent.fast_reward_model.load_state_dict(torch.load(fast_reward_model_path, map_location='cpu'))
        with open(reward_norm_path, 'rb') as f:
            obs_rms =  pickle.load(f)

        print('load finished!')

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done, save_img=save_image_dir)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    if 'gymveccont' not in args['EnvType']:
        if 'vec' in args['EnvType']:
            states = np.zeros([num_worker, input_size[0]])
        else:
            states = np.zeros([num_worker, 4, 84, 84])
    else:
        s0 = work.reset()
        states = np.repeat(np.expand_dims(s0, axis=0), num_worker, axis=0)
    print(states.shape)

    if args['Mode']=='train':
        writer = SummaryWriter(logdir=logdir)
        with open(param_dir, 'w') as fp:
            json.dump(args, fp)

        sample_episode = 0
        sample_rall = 0
        sample_step = 0
        sample_env_idx = 0
        sample_i_rall = 0
        global_update = 0
        global_step = 0

        if args['UseScaler']:
            scaler = init_scaler(env, agent, input_size[0], num=5)

        # normalize obs
        print('Start to initailize observation normalization parameter.....')
        next_obs = []
        states0 = states
        for step in tqdm(range(num_step * pre_obs_norm_step)):

            if 'gymveccont' == args['EnvType'] or 'duckie' in args['EnvID'].lower():
                actions, _,_,_ = agent.get_action(states0)
            else:
                actions = np.random.randint(0, agent.output_size, size=(num_worker,))
            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_states=[]
            for parent_conn in parent_conns:
                s, r, d, rd, lr = parent_conn.recv()
                if args['UseScaler']:
                    s = scaler.normalize_obs(s)
                next_states.append(s)
                if 'vec' in args['EnvType'] or (args['TrainMethod'] != 'RND' and args['TrainMethod'] !='RND_SM'):
                    next_obs.append(s)
                else:
                    next_obs.append(s[3, :, :].reshape([1, 84, 84]))
            states0 = np.stack(next_states)
            next_states = []



            if len(next_obs) % (num_step * num_worker) == 0:
                next_obs = np.stack(next_obs)
                obs_rms.update(next_obs)
                next_obs = []

        print('End to initalize...')
        pbar = tqdm(total=args['MaxEnvStep'])
        all_returns = deque(maxlen=200)
        best_return = -10000000000

        while global_step<args['MaxEnvStep']:
            total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy, total_policy_np = \
                [], [], [], [], [], [], [], [], [], [], []
            pbar.update(num_worker * num_step)
            global_step += (num_worker * num_step)
            global_update += 1
            agent.global_step = global_step
            # Step 1. n-step rollout

            for _ in range(num_step):

                actions, value_ext, value_int, policy = agent.get_action(np.float32(states))
                if 'cont' not in args['EnvType']:
                    factions = actions.squeeze(-1)
                else:
                    factions = actions
                for parent_conn, action in zip(parent_conns, factions):
                    parent_conn.send(action)

                next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
                for parent_conn in parent_conns:
                    s, r, d, rd, lr = parent_conn.recv()
                    if args['UseScaler']:
                        s = scaler.normalize_obs(s)
                    next_states.append(s)
                    rewards.append(r)
                    dones.append(d)
                    real_dones.append(rd)
                    log_rewards.append(lr)
                    if 'vec' in args['EnvType'] or (args['TrainMethod'] != 'RND' and args['TrainMethod'] !='RND_SM'):
                        next_obs.append(s)
                    else:
                        next_obs.append(s[3, :, :].reshape([1, 84, 84]))

               


                next_states = np.stack(next_states)
                rewards = np.hstack(rewards)
                dones = np.hstack(dones)
                real_dones = np.hstack(real_dones)
                next_obs = np.stack(next_obs)

                if args['TrainMethod'] == 'RND_SM':
                    if "noM" not in args['MemType']:  
                        if len(agent.M_memory.memory_list)==0:
                            res = np.zeros([dones.shape[0],512])
                            agent.update_M_memory(res, dones)
                            intrinsic_reward, fast_reward, res, _ = agent.compute_intrinsic_reward(
                                ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))
                        else:
                            intrinsic_reward, fast_reward, res, _ = agent.compute_intrinsic_reward(
                                ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))
                            agent.update_M_memory(res.cpu(), dones)
                    else:
                        intrinsic_reward, fast_reward, res, _ = agent.compute_intrinsic_reward(
                            ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))
                if args['TrainMethod'] == 'RND':
                    intrinsic_reward = agent.compute_intrinsic_reward(
                        ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))
               
                intrinsic_reward = np.hstack(intrinsic_reward)
                sample_i_rall += intrinsic_reward[sample_env_idx]
                total_next_obs.append(next_obs)
                total_int_reward.append(intrinsic_reward)
                total_state.append(states)
                total_reward.append(rewards)
                total_done.append(dones)
                total_action.append(actions)
                total_ext_values.append(value_ext)
                total_int_values.append(value_int)
                total_policy.append(policy)
                total_policy_np.append(policy.cpu().numpy())

                states = next_states

                sample_rall += log_rewards[sample_env_idx]

                sample_step += 1
                if real_dones[sample_env_idx]:
                    sample_episode += 1
                    agent.log_state = []
                    agent.log_surp = []
                    agent.log_surp_novel = []

                    all_returns.append(sample_rall)

                    writer.add_scalar('episode/avg_return', np.mean(list(all_returns)), sample_episode)
                    writer.add_scalar('episode/sampled_return', sample_rall, sample_episode)
                    writer.add_scalar('data/avg_return', np.mean(list(all_returns)), global_step)
                    writer.add_scalar('data/sampled_return', sample_rall, global_step)

                    writer.add_scalar('episode/epi_step', sample_step, sample_episode)
                    writer.add_scalar('data/epi_step', sample_step, global_step)
                    writer.add_scalar('loss/policy loss', np.mean(list(agent.policy_losses)[-50:]), global_step)
                    writer.add_scalar('loss/reward loss', np.mean(list(agent.reward_losses)[-50:]), global_step)

                    if "RND_SM" in args["TrainMethod"]:
                        writer.add_scalar('data/M reward', np.mean(list(agent.fast_rewards)[-50:]), global_step)
                        writer.add_scalar('data/W reward', np.mean(list(agent.slow_rewards)[-50:]), global_step)


                    sample_rall = 0
                    sample_step = 0
                    sample_i_rall = 0

            # calculate last next value
            _, value_ext, value_int, _ = agent.get_action(np.float32(states))
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            # --------------------------------------------------

            if 'vec' in args['EnvType']:
                total_state = np.stack(total_state).transpose([1, 0, 2]).reshape([-1, input_size_dim])
                total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2]).reshape([-1, input_size_dim])
            else:
                total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
                total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, 84, 84])

            total_reward = np.stack(total_reward).transpose().clip(-1, 1)
            # print(np.stack(total_action))
            total_action = np.stack(total_action).transpose([1, 0, 2]).reshape([-1,actions.shape[-1]])
            total_done = np.stack(total_done).transpose()
            total_ext_values = np.stack(total_ext_values).transpose()
            total_int_values = np.stack(total_int_values).transpose()
            total_logging_policy = np.vstack(total_policy_np)

            # Step 2. calculate intrinsic reward
            # running mean intrinsic reward
            total_int_reward = np.stack(total_int_reward).transpose()
            total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                             total_int_reward.T])
            mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
            reward_rms.update_from_moments(mean, std ** 2, count)

            # normalize intrinsic reward
            total_int_reward /= np.sqrt(reward_rms.var)
            writer.add_scalar('episode/total_int_reward', np.sum(total_int_reward) / num_worker, sample_episode)
            writer.add_scalar('data/total_int_reward', np.sum(total_int_reward) / num_worker, global_step)
            # -------------------------------------------------------------------------------------------

            # logging Max action probability
            writer.add_scalar('episode/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

            # Step 3. make target and advantage
            # extrinsic reward calculate
            ext_target, ext_adv = make_train_data(total_reward,
                                                  total_done,
                                                  total_ext_values,
                                                  gamma,
                                                  num_step,
                                                  num_worker, args)

            # intrinsic reward calculate
            # None Episodic

            int_target, int_adv = make_train_data(total_int_reward,
                                                  np.zeros_like(total_int_reward),
                                                  total_int_values,
                                                  int_gamma,
                                                  num_step,
                                                  num_worker, args)

            # add ext adv and int adv
            total_adv = int_adv * int_coef + ext_adv * ext_coef
            # -----------------------------------------------

            # Step 4. update obs normalize param
            obs_rms.update(total_next_obs)
            # -----------------------------------------------

            # Step 5. Training!

            agent.train_model(np.float32(total_state), ext_target, int_target, total_action,
                              total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                              total_policy, args)



            if global_step % (num_worker * num_step)*1000 == 0:
                mean_return = np.mean(list(all_returns))
                print('Now Global Step :{}, num eps: {}, avg return: {}'.format(global_step, sample_episode, mean_return))
                if mean_return>best_return:
                    best_return = mean_return
                    torch.save(agent.model.state_dict(), model_path)
                    torch.save(agent.dist.state_dict(), dist_path)
                    torch.save(agent.reward_model.state_dict(), reward_model_path)
                    # if "AIM" in args['TrainMethod']:
                    #     torch.save(agent.fast_reward_model.state_dict(), fast_reward_model_path)
                    with open(reward_norm_path, 'wb') as f:
                        # Pickle the 'data' dictionary using the highest protocol available.
                        pickle.dump(obs_rms, f, pickle.HIGHEST_PROTOCOL)
                    print(f"SAVE TO {model_path}")

        pbar.close()

    elif args['Mode']=='test':
        print("----TEST----")
        args['NumEnv']=1
        if  args['ActionFile']:
            with  open(args['ActionFile'], "r") as afile:

                # reading the file
                data = afile.read()
                # replacing end splitting the text
                # when newline ('\n') is seen.
                eps_acts = data.split("---")
                file_actions = []
                for e in eps_acts:
                    file_actions.append(e.strip().split("\n"))
                print(file_actions)

        list_rewards = []
        for i in range(args['NumEval']):
            # if 'vec' in args['EnvType']:
            #     states = np.zeros([num_worker, input_size_dim])
            # else:
            #     states = np.zeros([num_worker, 4, 84, 84])
            steps = 0
            rall = 0
            rd = False
            intrinsic_reward_list = []
            fast_reward_list = []
            sreward_list = []
            sample_rall = 0


            while not rd:
                if not args['ActionFile']:
                    actions, value_ext, value_int, policy = agent.get_action(np.float32(states))
                else:
                    actions = [file_actions[i][steps%len(file_actions[i])]]

                steps += 1

                # print(actions)
                for parent_conn, action in zip(parent_conns, actions):
                    parent_conn.send(action)

                next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
                for parent_conn in parent_conns:
                    s, r, d, rd, lr = parent_conn.recv()
                    # if steps >180:
                    #     rd = True
                    next_states.append(s)
                    rewards.append(r)
                    dones.append(d)
                    real_dones.append(rd)
                    log_rewards.append(lr)
                    # print(r, lr)
                    if 'vec' in args['EnvType'] or (args['TrainMethod'] != 'RND' and args['TrainMethod'] !='AIM'):
                        next_obs.append(s)
                    else:
                        next_obs.append(s[3, :, :].reshape([1, 84, 84]))

                next_states = np.stack(next_states)
                rewards = np.hstack(rewards)
                dones = np.hstack(dones)
                real_dones = np.hstack(real_dones)
                next_obs = np.stack(next_obs)
                sample_rall += log_rewards[0]

                # total reward = int reward + ext Reward
                if args['TrainMethod'] == 'RND_SM':
                   
                    if "noM" not in args['MemType']: 
                        if len(agent.M_memory.memory_list)==0:
                            res = np.zeros([dones.shape[0],512])
                            agent.update_M_memory(res, dones)
                            intrinsic_reward, fast_reward, res, sreward = agent.compute_intrinsic_reward(
                                ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))
                        else:
                            intrinsic_reward, fast_reward, res, sreward = agent.compute_intrinsic_reward(
                                ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))
                            agent.update_M_memory(res.cpu(), dones)
                    else:
                        intrinsic_reward, fast_reward, res, sreward = agent.compute_intrinsic_reward(
                            ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))
                    fast_reward_list.append(fast_reward)
                    sreward_list.append(sreward)
                if args['TrainMethod'] == 'RND':
                    intrinsic_reward = agent.compute_intrinsic_reward(
                        ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))

                intrinsic_reward_list.append(intrinsic_reward)

                states = next_states


                if rd:
                    intrinsic_reward_list = (intrinsic_reward_list - np.mean(intrinsic_reward_list)) / np.std(
                        intrinsic_reward_list)
                    sreward_list = (sreward_list - np.mean(sreward_list)) / np.std(
                        sreward_list)
                    print(sample_rall)
                    list_rewards.append(sample_rall)
                    c=1
                    c2=1
                    c3=1
                    irlist = []
                    srlist = []
                    with open(save_image_dir+'/ir.txt', 'w') as f:
                        for ir in intrinsic_reward_list:
                            if len(ir.shape)==1:
                                f.write(str(c)+": "+str(ir[0]))
                                irlist.append(ir[0])
                            else:
                                f.write(str(c)+": "+str(ir[0][0]))
                                irlist.append(ir[0][0])
                            f.write('\r\n')
                            c+=1
                    with open(save_image_dir+'/fr.txt', 'w') as f:
                        for ir in fast_reward_list:
                            f.write(str(c2)+": "+str(ir[0][0]))
                            f.write('\r\n')
                            c2+=1
                    with open(save_image_dir+'/sr.txt', 'w') as f:
                        for ir in sreward_list:
                            if len(ir.shape)==1:
                                f.write(str(c3)+": "+str(ir[0]))
                                srlist.append(ir[0])
                            else:
                                f.write(str(c3)+": "+str(ir[0][0]))
                                srlist.append(ir[0][0])
                            f.write('\r\n')

                            c3+=1

                    steps = 0
                    rall = 0
                    sample_rall=0

        print(f"{np.mean(list_rewards)}+-{np.std(list_rewards)}")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = ArgumentParser(description="Training script for RND+PPO.")
    parser.add_argument("--Mode", type=str, default="train",
                        help="train or test")
    parser.add_argument("--NumEval", type=int, default=1,
                        help="item size")
    parser.add_argument("--MemType", type=str, default="sum",
                        help="full/noM/noW")
    parser.add_argument("--MemNoise", type=float, default=0,
                        help="noise query mem")
    parser.add_argument("--MemSize", type=int, default=16,
                        help="item size")
    parser.add_argument("--MemLength", type=int, default=128,
                        help="N")
    parser.add_argument("--FeatureSize", type=int, default=512,
                        help="feature size")
    parser.add_argument("--load_path", type=str, default="",
                        help="load model from path?")
    parser.add_argument("--savedir", type=str, default="./models",
                        help="save model to dir")
    parser.add_argument("--logdir", type=str, default="./runs",
                        help="root log dir")
    parser.add_argument("--TrainMethod", type=str, default="RND",
                        help="type of curiosity")
    parser.add_argument("--EnvType", type=str, default="atari",
                        help="atari/mario/classic")
    parser.add_argument("--EnvID", default="MontezumaRevengeNoFrameskip-v4",
                        help="name of env")
    parser.add_argument("--ExtCoef", type=float, default=2,
                        help="extrinsic coef")
    parser.add_argument("--IntCoef", type=float, default=1,
                        help="intrinsic coef")
    parser.add_argument("--LearningRate", type=float, default=1e-4,
                        help="lr")
    parser.add_argument("--MaxEnvStep", type=int, default=1e8,
                        help="Nstep")
    parser.add_argument("--NumStep", type=int, default=128,
                        help="Nstep")
    parser.add_argument("--NumEnv", type=int, default=128,
                        help="num workers")
    parser.add_argument("--Gamma", type=float, default=0.999,
                        help="discounted factor for extrinsic reward")
    parser.add_argument("--IntGamma", type=float, default=0.99,
                        help="discounted factor for intrinsic reward")
    parser.add_argument("--Lambda", type=float,default=0.95,
                        help="GAE lambda")
    parser.add_argument("--StableEps", type=float, default=1e-8,
                        help="epsilon")
    parser.add_argument("--StateStackSize", type=int, default=4,
                        help="num frames stacked")
    parser.add_argument("--UseGAE", type=str2bool, default=True,
                        help="enable GAE?")
    parser.add_argument("--UseGPU", type=str2bool, default=True,
                        help="enable GPU")
    parser.add_argument("--UseNorm", type=str2bool, default=False,
                        help="normalizaiton")
    parser.add_argument("--UseNoisyNet", type=str2bool, default=False,
                        help="enable Noisy Net explore")
    parser.add_argument("--ClipGradNorm", type=float, default=0.5,
                        help="clip grad")
    parser.add_argument("--Entropy", type=float, default= 0.001,
                        help="entropy coef")
    parser.add_argument("--Epoch", type=int, default=4,
                        help="ppo epochs")
    parser.add_argument("--MiniBatch", type=int, default=4,
                        help="ppo batch size")
    parser.add_argument("--PPOEps", type=float, default=0.2,
                        help="ppo epsilon")
    parser.add_argument("--StickyAction", type=str2bool, default=True,
                        help="enable stickky action")
    parser.add_argument("--ActionProb", type=float, default=0.25,
                        help="sticky action prob")
    parser.add_argument("--UpdateProportion", type=float, default=0.25,
                        help="entropy coef")
    parser.add_argument("--LifeDone", type=str2bool, default=False,
                        help="done life end")
    parser.add_argument("--ObsNormStep", type=int, default=50,
                        help="num steps normalizing observations")
    parser.add_argument("--UseScaler", type=int, default=0,
                        help="scale the state")
    parser.add_argument("--Seed", type=int, default=0,
                        help="seed run")

    args  = parser.parse_args()
    print(args.TrainMethod)
    if args.TrainMethod not in ['RND', 'RND_SM']:
        args.TrainMethod = 'RND'
        args.IntCoef=0
    if 'gymvec' in args.EnvType:
        print('vector state space env ...')
        args.ObsNormStep=5
    else:
        print('image state space env ...')

    main(args)
