import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch import optim
import os

import math

# the defination of the hyper parameters
BATCH_SIZE = 32  # batch size of the training data
LR = 0.001  # learning rate
EPSILON = 0.6  # greedy algorithm
GAMMA = 0.9  # reward discount
TARGET_UPDATE = 100  # update the target network after training
MEMORY_CAPACITY = 10000  # the capacity of the memory
N_STATE = 4  # the number of states that can be observed from environment
N_ACTION = 2  # the number of action that the agent should have
# decide the device used to train the network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# the structure of the network

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 4000

class Net(nn.Module):
    def __init__(self, n_state=N_STATE, n_action=N_ACTION):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_state, 64)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(64, n_action)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = x.to(device)
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out


class DQN(object):
    def __init__(self, test=False, var_eps = True):
        self.policy_net, self.target_net = Net().to(device), Net().to(device)
        self.learn_step = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATE*2 + 2))
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.eval_model_load_path = './model_eval.pt'
        self.target_model_load_path = './model_target.pt'
        self.eval_model_save_path = './model_evalV2.pt'
        self.target_model_save_path = './model_targetV2.pt'
        self.test_mode = test
        self.var_eps = var_eps

    def select_action(self, state):

        state = torch.unsqueeze(torch.tensor(state), dim = 0)
        p = np.random.random()

        if os.path.exists(self.eval_model_load_path):
            E_thresh = EPS_END
        else:
            E_thresh = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.learn_step / EPS_DECAY)
        if self.test_mode:
            actions_value = self.policy_net.forward(state)
            return torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        else:
            if p > E_thresh:
                actions_value = self.policy_net.forward(state)
                return torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
            else:
                return np.random.randint(0, N_ACTION)


    def store_transition(self, s, a ,r, s_):
        transition = np.hstack((s, a, r, s_))

        # 这里的设计非常有意思，实现了memory的自动替换
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index,:] = transition
        self.memory_counter += 1

    def optimize_model(self):

        if self.learn_step % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.learn_step += 1

        # get the samples to train the policy net
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        sample_batch = self.memory[sample_index, :]
        batch_s = torch.FloatTensor(sample_batch[:, :N_STATE]).to(device)
        batch_a = torch.LongTensor(sample_batch[:, N_STATE:N_STATE+1].astype(int)).to(device)
        batch_r = torch.FloatTensor(sample_batch[:, N_STATE+1:N_STATE+2]).to(device)
        batch_s_ = torch.FloatTensor(sample_batch[:, -N_STATE:]).to(device)

        #calculate the q_value
        q_eval = self.policy_net(batch_s).gather(1, batch_a)
        q_next = self.target_net(batch_s_).max(1)[0].detach()  # use detach to avoid the backpropagation during the training
        q_traget = batch_r.squeeze() + GAMMA*q_next
        loss = self.loss_func(q_eval.squeeze(), q_traget)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.eval_model_save_path)
        torch.save(self.target_net.state_dict(), self.target_model_save_path)

    def load_model(self):
        self.policy_net.load_state_dict(torch.load(self.eval_model_load_path))
        self.target_net.load_state_dict(torch.load(self.target_model_load_path))






















