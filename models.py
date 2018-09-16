# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 10:03:09 2018

@author: CHEUNMI2
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 12)
        self.fc1.weight.data.normal(0, 0.1)
        self.fc2 = nn.Linear(12, 12)
        self.fc2.weight.data.normal(0, 0.1)
        self.out = nn.Linear(12, n_actions)
        self.out.weight.data.normal(0, 0.1)
        
        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            actions_value = self.out(x)
            return actions_value

class DQN():
    def __init__(self, epsilon, learning_rate, gamma, batch_size, target_replace_iter, memory_capacity, n_actions, n_states):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity
        self.n_actions = n_actions
        self.n_states = n_states
        
        self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
    
    def get_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        
        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0,0]
        else:
            action = np.random.randint(0, self.n_actions)
            
        return action
    
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1
        
    def learn(self):
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_states])
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:])
        
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0]
        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()