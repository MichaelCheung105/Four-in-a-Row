import numpy as np
from models import model

class agent():
    def __init__(self, role: int, algo: str, epsilon: float, learning_rate: float):
        self.role = role
        self.model = model(algo, epsilon, learning_rate)

    def step(self, board: np.ndarray):
        action = self.model.get_action(board)
        location = 5-(np.fliplr(board.T)==0).argmax(axis=1)[action]
        
        while board[location, action] != 0:
            print('Occupied!! Try another move')
            action = self.model.get_action(board)
            location = 5-(np.fliplr(board.T)==0).argmax(axis=1)[action]
            
        board[location,action] = self.role
        return board

    def learn(self, winner: int):
        if winner == 0:
            self.model.bp(0)
        
        elif self.role == winner:
            self.model.bp(1)

        else:
            self.model.bp(-1)
            
