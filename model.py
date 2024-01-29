import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init
from collections import deque
from random import randint

# Batched index_select
def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy) # b x e x f
    return out

class M_Memory(nn.Module):

    def __init__(self, batch_size, T, feature_size, item_size, device, total_time=1e8):
        super().__init__()
        self.item_size = item_size
        self.batch_size = batch_size
        self.T = 128
        self.device = device
        self.M = torch.Tensor(batch_size, self.T, item_size).to(device=device)
        self.M2 = torch.Tensor(batch_size, self.T, item_size).to(device=device)
        self.Ind = torch.Tensor(batch_size, self.T).to(device=device)
        self.Ind2 = torch.Tensor(batch_size, self.T).to(device=device)
        self.BM=None
        self.memory_list = deque(maxlen=T)
        self.reset_memory()
        self.total_time = total_time
        self.coef=0.01

    def clone_mem(self):
        return self.M.detach().clone()

    def reset_memory(self):
        self.M.data.zero_()
        self.Ind.data.zero_()
        self.Ind[:,0] = 1

    def init_memory(self):
        stdv = 1. / math.sqrt(self.M.size(1))
        return torch.FloatTensor(self.batch_size, self.T, self.item_size).zero_().to(device=self.device)
        # return torch.zeros(self.batch_size, self.item_size, self.item_size).to(device=self.device)

    def init_Ind(self):
        X = torch.Tensor(batch_size, T).to(device=device)
        X[:, 0]=1
        return X

    def add_item(self, item, done):
        '''
        item: shape=Bxd
        done: shape=Bx1
        '''
        self.memory_list.append(self.clone_mem())
        item = self.normalize(item.detach())
        self.M += torch.matmul(self.Ind.unsqueeze(2), item.unsqueeze(1))
        self.Ind = torch.roll(self.Ind, 1, 1)
        self.M = self.M*(1.0-done.unsqueeze(-1).unsqueeze(-1))+self.init_memory()*done.unsqueeze(-1).unsqueeze(-1)

    def collect_memory(self):
        self.BM = torch.stack(list(self.memory_list), dim=1).to(device=self.device).reshape(-1, self.item_size, self.item_size)

    def normalize(self, q):
        return q
        qn = torch.norm(q, p=2, dim=-1).detach()
        q = q/qn.unsqueeze(-1)
        q[torch.isnan(q)]=0
        return q

    """
    FOR INFERENCE
    """
    def retrieve(self, query, noise=0):
        '''
        query: shape=Bxd
        return: shape=Bx1, shape=Bxd, Bxd
        '''
        query = self.normalize(query)

        A = self.memory_list[-1] #BxTxd
        AT = torch.transpose(A, 1, 2)# BxdXT
        address = torch.matmul(A, query.unsqueeze(-1)) #BxTx1
        address = address.squeeze(-1) #BxT
    
        address = torch.softmax(address, dim=-1).unsqueeze(-1) #Bx5x1
        recall = torch.matmul(AT, address) #Bxdx1
        recall = recall.squeeze(-1)
    
        return self.coef*torch.mean((recall-query).pow(2), dim=-1, keepdim=True), (recall-query).detach(), query.detach()


    """
    FOR TRAINING
    """
    def retrieve_item(self, query, indices, noise=0):
        '''
        query: shape=B*Nxd
        indices: shape=N
        return: shape=B*Nxd
        '''
        M = torch.index_select(self.BM, 0, indices) #B*NxTxd

        A = M #B*NxTxd
        AT = torch.transpose(M, 1, 2)# B*NxdXT
        address = torch.matmul(A,query.unsqueeze(-1)) #B*NxTx1
        address = address.squeeze(-1) #B*NxT
     
        address = torch.softmax(address, dim=-1).unsqueeze(-1) #B*Nx5x1
        recall = torch.matmul(AT, address) #B*Nxdx1

        return recall.squeeze(-1)

    def retrieve_train(self, item, indices, noise=0):
        '''
        input_tensor: shape=B*Nxd
        indices: shape=N
        return: shape=B*Nx1, shape=B*Nxd, B*nxd
        '''
        query = self.normalize(item)
        recall = self.retrieve_item(query, indices, noise) #B*Nxd
        return recall, recall-query.detach(), query.detach()


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet"""

    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.noisy_weight = nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.noisy_bias = nn.Parameter(torch.Tensor(out_features))
        self.noise_std = sigma0 / math.sqrt(self.in_features)

        self.reset_parameters()
        self.register_noise()

    def register_noise(self):
        in_noise = torch.FloatTensor(self.in_features)
        out_noise = torch.FloatTensor(self.out_features)
        noise = torch.FloatTensor(self.out_features, self.in_features)
        self.register_buffer('in_noise', in_noise)
        self.register_buffer('out_noise', out_noise)
        self.register_buffer('noise', noise)

    def sample_noise(self):
        self.in_noise = torch.normal(torch.zeros(self.in_features), torch.ones(self.in_features)*self.noise_std)
        self.out_noise = torch.normal(torch.zeros(self.out_features), torch.ones(self.out_features)*self.noise_std)
        if torch.cuda.is_available():
            self.in_noise = self.in_noise.cuda()
            self.out_noise = self.out_noise.cuda()
        self.noise = torch.mm(
            self.out_noise.view(-1, 1), self.in_noise.view(1, -1))



    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.noisy_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.noisy_bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        Note: noise will be updated if x is not volatile
        """
        normal_y = nn.functional.linear(x, self.weight, self.bias)
        if self.training:
            # update the noise once per update
            self.sample_noise()

        noisy_weight = self.noisy_weight * self.noise
        noisy_bias = self.noisy_bias * self.out_noise
        noisy_y = nn.functional.linear(x, noisy_weight, noisy_bias)
        return noisy_y + normal_y

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, use_noisy_net=False):
        super(CnnActorCriticNetwork, self).__init__()

        if use_noisy_net:
            print('use NoisyNet')
            linear = NoisyLinear
        else:
            linear = nn.Linear

        if len(input_size)==1:
            self.feature = nn.Sequential(
            nn.Linear(input_size[0], 512), nn.Tanh(),
            nn.Linear(512, 448), nn.Tanh())
        else:
            self.feature = nn.Sequential(
                nn.Conv2d(
                    in_channels=4,
                    out_channels=32,
                    kernel_size=8,
                    stride=4),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=4,
                    stride=2),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1),
                nn.ReLU(),
                Flatten(),
                linear(
                    7 * 7 * 64,
                    256),
                nn.ReLU(),
                linear(
                    256,
                    448),
                nn.ReLU()
            )

        self.actor = nn.Sequential(
            linear(448, 448),
            nn.ReLU(),
            linear(448, output_size)
        )

        self.extra_layer = nn.Sequential(
            linear(448, 448),
            nn.ReLU()
        )

        self.critic_ext = linear(448, 1)
        self.critic_int = linear(448, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        init.orthogonal_(self.critic_ext.weight, 0.01)
        self.critic_ext.bias.data.zero_()

        init.orthogonal_(self.critic_int.weight, 0.01)
        self.critic_int.bias.data.zero_()

        for i in range(len(self.actor)):
            if type(self.actor[i]) == nn.Linear:
                init.orthogonal_(self.actor[i].weight, 0.01)
                self.actor[i].bias.data.zero_()

        for i in range(len(self.extra_layer)):
            if type(self.extra_layer[i]) == nn.Linear:
                init.orthogonal_(self.extra_layer[i].weight, 0.1)
                self.extra_layer[i].bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value_ext = self.critic_ext(self.extra_layer(x) + x)
        value_int = self.critic_int(self.extra_layer(x) + x)
        return policy, value_ext, value_int


class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNDModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        if len(input_size) == 1:
            self.predictor = nn.Sequential(
                nn.Linear(input_size[0], 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512)
            )

            self.target = nn.Sequential(
                nn.Linear(input_size[0], 512),
                nn.ReLU(),
                nn.Linear(512, 512)
            )

        else:
            feature_output = 7 * 7 * 64
            self.predictor = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=8,
                    stride=4),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=4,
                    stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1),
                nn.LeakyReLU(),
                Flatten(),
                nn.Linear(feature_output, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512)
            )

            self.target = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=8,
                    stride=4),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=4,
                    stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1),
                nn.LeakyReLU(),
                Flatten(),
                nn.Linear(feature_output, 512)
            )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False


    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature



class RND_SM_Model(nn.Module):
    def __init__(self, input_size, output_size, item_size):
        super(RND_SM_Model, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.predictor = nn.Sequential(
                nn.ReLU(),
                nn.Linear(512, 512))
      

        if len(input_size) == 1:


            self.feature = nn.Sequential(
                nn.Linear(input_size[0], 512),
                nn.ReLU(),
                nn.Linear(512, 512),
            )

            self.target = nn.Sequential(
                nn.Linear(input_size[0], 512),
                nn.ReLU(),
                nn.Linear(512, 512)
            )

        else:
            feature_output = 7 * 7 * 64
            self.feature = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=8,
                    stride=4),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=4,
                    stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1),
                nn.LeakyReLU(),
                Flatten(),
                nn.Linear(feature_output, 512),
                nn.ReLU(),
                nn.Linear(512, 512)
            )

            self.target = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=8,
                    stride=4),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=4,
                    stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1),
                nn.LeakyReLU(),
                Flatten(),
                nn.Linear(feature_output, 512)
            )

        self.W_memory = nn.Sequential(
            nn.Linear(512+item_size, 32),
            nn.Tanh(),
             nn.Linear(32, 512+item_size)
        )
        


        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False



    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(self.feature(next_obs))

        return predict_feature, target_feature

    def freeze_feature(self):
        for param in self.feature.parameters():
            param.requires_grad = False
        self.is_freezed = True

    def get_feature(self, next_obs):
        feature = self.feature(next_obs)
        return feature

    def get_feature_fix(self, next_obs):
        feature = self.target(next_obs)
        return feature

    def predict_feature(self, feature):
        predict_feature = self.predictor(feature)
        return predict_feature

    def predict_residual(self, res):
        out = self.W_memory(res)
        return out

    def get_surprise_novelty(self, res, noise=0):
        res = torch.nn.functional.dropout(res, p=noise)
        o1 = self.W_memory(res.detach())
        o2 = res.clone()
        return o1- o2.detach(), [o1, o2]




