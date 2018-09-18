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
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(12, 12)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(12, n_actions)
        self.out.weight.data.normal_(0, 0.1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self, total_episode, epsilon, learning_rate, gamma, batch_size, target_replace_iter, memory_capacity, n_actions, n_states):
        self.total_episode = total_episode
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

    def get_action(self, board, episode, available):
        board = torch.unsqueeze(torch.FloatTensor(board), 0)

        self.epsilon = episode / self.total_episode
        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net.forward(board)
            sequence = np.argsort(actions_value.detach().numpy()[0])
            action = sequence[available[sequence]][-1]
        else:
            action = np.random.choice(np.array(range(self.n_actions))[available])

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
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()