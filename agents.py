import numpy as np
from models import DQN

class agent():
    def __init__(self, role, total_episode, epsilon, learning_rate, gamma, batch_size, target_replace_iter, memory_capacity, n_actions, n_states):
        self.role = role
        self.model = DQN(total_episode, epsilon, learning_rate, gamma, batch_size, target_replace_iter, memory_capacity, n_actions, n_states)

    def step(self, board: np.ndarray, episode):
        available = np.array([0 in board[:,col] for col in range(7)])
        action = self.model.get_action(board.flatten(), episode, available)
        location = 5-(np.fliplr(board.T)==0).argmax(axis=1)[action]
        
        while board[location, action] != 0:
            print('Occupied!! Try another move')
            available[action] = False
            action = self.model.get_action(board.flatten(), episode, available)
            location = 5-(np.fliplr(board.T)==0).argmax(axis=1)[action]
            
        board[location,action] = self.role
        return board, action
    
    def store(self, in_board, action, winner, board):
        s  = in_board.flatten()
        a  = action
        r  = winner * self.role
        s_ = board.flatten()
        self.model.store_transition(s, a, r, s_)
        
    def random_action(self, board: np.ndarray):
        available = np.array([0 in board[:,col] for col in range(7)])
        action = np.random.choice(np.array(range(7))[available])
        location = 5-(np.fliplr(board.T)==0).argmax(axis=1)[action]
        
        while board[location, action] != 0:
            print('Occupied!! Try another move')
            action = self.model.get_action(board)
            location = 5-(np.fliplr(board.T)==0).argmax(axis=1)[action]
            
        board[location,action] = self.role
        return board, action