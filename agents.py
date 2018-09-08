import numpy as np
from util import score

class agent():
    def __init__(self, role: int, model: object):
        self.role = role
        self.model = model
        self.memory = []

    def place_chess(self, board: np.ndarray):
        winner = score(board)
        return board, winner

    def bp(self, winner: int):
        if self.role == 1 and winner == 1:
            pass

        if self.role == 1 and winner == -1:
            pass

        if self.role == -1 and winner == 1:
            pass

        if self.role == -1 and winner == -1
            pass