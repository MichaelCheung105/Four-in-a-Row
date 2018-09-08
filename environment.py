import numpy as np
import time
import pickle

class environment():
    def __init__(self, first_mover: object, second_mover: object):
        self.first_mover = first_mover
        self.second_mover = second_mover

    def reset(self):
        return np.zeros(shape=(6,7), dtype=int)

    def train(self, total_episode: int, epsilon: int, learning_rate: int):
        episode = 0
        while episode < total_episode:
            episode += 1
            board = self.reset()
            winner = 0
            start_time = time.time()
            while 0 in board or winner != 0:
                board, winner = self.first_mover.place_chess(board, epsilon, learning_rate)
                board, winner = self.second_mover.place_chess(board, epsilon, learning_rate)
            self.first_mover.bp(winner=winner)
            self.second_mover.bp(winner=winner)
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