import numpy as np
import time
import pickle
from util import score

class environment():
    # initialize the 2 computer agents
    def __init__(self, first_mover: object, second_mover: object):
        self.first_mover = first_mover
        self.second_mover = second_mover

    # reset the chess board
    def reset(self):
        return np.zeros(shape=(6,7), dtype=int)

    # train the 2 computer agents
    def train(self, total_episode: int, epsilon: int, learning_rate: int):
        episode = 0
        while episode < total_episode:
            # reset the episode
            episode += 1
            board = self.reset()
            winner = 0
            start_time = time.time()
            
            # 2 agents keep playing against each other until a winner is decided or there is no more space for placing chess
            while 0 in board and winner == 0:
                board = self.first_mover.step(board)
                winner = score(board)
                if winner != 0:
                    break
                
                elif 0 in board:
                    board = self.second_mover.step(board)
                    winner = score(board)
                    if winner != 0:
                        break
                    
            # train the DQN
            self.first_mover.learn(winner=winner)
            self.second_mover.learn(winner=winner)
            
            end_time = time.time()
            print("Episode " + str(episode) + " (" + str(np.round(end_time-start_time)) + " seconds)")

        pickle.dump(self.first_mover, "./trained_models/first_mover.p")
        pickle.dump(self.second_mover, "./trained_models/second_mover.p")

    def test(self, trained_agent: str):
        if trained_agent == 'first_mover':
            pass

        elif trained_agent == 'second_mover':
            pass

        elif trained_agent == 'both agents':
            pass

        else:
            print('please either pass "first_mover", "second_mover" or "both agents" as argument')

    def com_as_player_1(self):
        pass

    def com_as_player_2(self):
        pass