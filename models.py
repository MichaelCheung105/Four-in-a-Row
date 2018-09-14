# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 10:03:09 2018

@author: CHEUNMI2
"""
import numpy as np

class model():
    def __init__(self, algo: str, epsilon: float, learning_rate: float):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        
        if algo == 'DQN':
            pass
        
        else:
            print('error')
    
    def get_action(self, board):
        action = np.random.choice(range(7))
        return action        
        
    def bp(self, category: int):
        pass

        
        
        
            
"""
# Building DQN
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d()
"""
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

#Hyper Parameters
batch_size = 32
learning_rate = 0.1
epsilon = 0.1
gamma = 0.9
target_replace_iter = 100
memory_capacity = 1000
n_actions = 7
n_states = 42

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 10)
        self.fc1.weight.data.normal(0, 0.1)
        self.out = nn.Linear(10, n_actions)
        self.out.weight.data.normal(0, 0.1)
        
        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            actions_value = self.out(x)
            return actions_value

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2))
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
    
    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if np.random.uniform() < epsilon:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0,0]
        else:
            action = np.random.randint(0, n_actions)
            
        return action
    
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1
        
    def learn(self):
        if self.learn_step_counter % target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        
        sample_index = np.random.choice(memory_capacity, batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :n_states]))
        b_a = Variable(torch.LongTensor(b_memory[:, n_states:n_states+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, n_states+1:n_states+2]))
        b_s = Variable(torch.FloatTensor(b_memory[:, -n_states:]))
        
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + gamma * q_next.max(1)[0]
        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()