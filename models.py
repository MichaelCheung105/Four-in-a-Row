# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 10:03:09 2018

@author: CHEUNMI2
"""

class model():
    def __init__(self, algo: str, epsilon: float, learning_rate: float):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        
        if algo == 'DQN':
            pass
        
        else:
            print('error')
    
    def get_action(self, board):
        processed_board = board.flatten()
        action = get(processed_board)
        return action        
        
    def bp(self, category: int):
        pass
        
        self.memory = []