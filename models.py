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