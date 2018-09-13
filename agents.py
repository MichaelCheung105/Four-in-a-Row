import numpy as np
from util import place_chess
from models import model

class agent():
    def __init__(self, role: int, algo: str, epsilon: float, learning_rate: float):
        self.role = role
        self.model = model(algo, epsilon, learning_rate)

    def step(self, board: np.ndarray):
        action = self.model.get_action(board)
        board = place_chess(board, action, self.role)
        return board

    def learn(self, winner: int):
        if winner == 0:
            self.model.bp(0)
        
        elif self.role == winner:
            self.model.bp(1)

        else:
            self.model.bp(-1)
            
            
            
            
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