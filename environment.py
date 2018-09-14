import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

class environment():
    # initialize the 2 computer agents
    def __init__(self, first_mover: object, second_mover: object):
        self.first_mover = first_mover
        self.second_mover = second_mover

    # reset the chess board
    def reset(self):
        return np.zeros(shape=(6,7), dtype=int)
    
    # function to calculate who win the game
    def score(self, board: np.ndarray, num=4):
        for row in range(board.shape[0]-num+1):
            for col in range(board.shape[1]-num+1):
                sub_board = board[row:row+num, col:col+num]
                checklist = [list(sub_board.sum(axis=0)), list(sub_board.sum(axis=1)), [sub_board.trace()], [np.fliplr(sub_board).trace()]]
                for rule in checklist:
                    if 4 in rule:
                        return 1
                    elif -4 in rule:
                        return -1
        return 0

    # train the 2 computer agents
    def train(self, total_episode: int):
        self.memory = {}
        episode = 0
        start_time = time.time()
        while episode < total_episode:
            # reset the episode
            episode += 1
            self.memory[episode] = []
            board = self.reset()
            winner = 0
            
            # 2 agents keep playing against each other until a winner is decided or there is no more space for placing chess
            while 0 in board and winner == 0:
                board = self.first_mover.step(board)
                winner = self.score(board)
                self.memory[episode].append(board.copy())
                if winner != 0:
                    break
                
                elif 0 in board:
                    board = self.second_mover.step(board)
                    winner = self.score(board)
                    self.memory[episode].append(board.copy())
                    if winner != 0:
                        break
            
            print("Episode " + str(episode))
            if winner == 0:
                print("It's a draw!")
            elif winner == 1:
                print('first_mover wins!!!')
            elif winner == -1:
                print('second_mover wins!!!')
                
            self.plot(board)
                
            # train the DQN
            self.first_mover.learn(winner=winner)
            self.second_mover.learn(winner=winner)

        end_time = time.time()
        print('training time taken (' + str(end_time - start_time) + "seconds)")
        
        """
        pickle.dump(self.first_mover, "./trained_models/first_mover.p")
        pickle.dump(self.second_mover, "./trained_models/second_mover.p")
        """

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
    
    def plot(self, board: np.ndarray):
        plt.figure()
        plt.imshow(board)
        plt.gca().set_xticks(np.arange(-0.5,6,1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5,5,1), minor=True)
        plt.gca().grid(which='minor', color='k', linestyle='-', linewidth=2)
        plt.show()
        plt.close()