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
        winning_record = []
        episode = 0
        start_time = time.time()
        while episode < total_episode:
            # reset the episode
            episode += 1
            print(episode)
            episode_memory = []
            board = self.reset()
            winner = 0
            
            # 2 agents keep playing against each other until a winner is decided or there is no more space for placing chess
            while 0 in board and winner == 0:
                in_board = board
                board, action = self.first_mover.step(in_board)
                winner = self.score(board)
                episode_memory.append({'in_board':in_board, 'action':action, 'winner':winner, 'board':board})
                
                if 0 in board and winner == 0:
                    in_board = board
                    board, action = self.second_mover.step(in_board)
                    winner = self.score(board)
                    episode_memory.append({'in_board':in_board, 'action':action, 'winner':winner, 'board':board})
                
            # print winner, store winner and plot trend
            '''
            print("Episode " + str(episode))
            if winner == 0:
                print("It's a draw!")
            elif winner == 1:
                print('first_mover wins!!!')
            elif winner == -1:
                print('second_mover wins!!!')
            '''
                
            winning_record.append(winner)
            if len(winning_record) % 100 == 0:
                print(winning_record.count(1))
                print(winning_record.count(0))
                print(winning_record.count(-1))
            
            # store transition
            episode_length = len(episode_memory)
            for index, memory in enumerate(episode_memory):
                if index % 2 == 0:
                    in_board = episode_memory[index]['in_board']
                    action = episode_memory[index]['action']
                    winner = episode_memory[index]['winner']                    
                    board = [episode_memory[index]['board'] if index + 1 == episode_length else episode_memory[index+1]['board']][0]
                    self.first_mover.store(in_board, action, winner, board)
                    
                else:
                    in_board = episode_memory[index]['in_board']
                    action = episode_memory[index]['action']
                    winner = episode_memory[index]['winner']
                    board = [episode_memory[index]['board'] if index + 1 == episode_length else episode_memory[index+1]['board']][0]
                    self.second_mover.store(in_board, action, winner, board)

            # train the DQN
            if self.first_mover.model.memory_counter > self.first_mover.model.memory_capacity:
                self.first_mover.model.learn()
            if self.second_mover.model.memory_counter > self.second_mover.model.memory_capacity:
                self.second_mover.model.learn()

        end_time = time.time()
        print('training time taken (' + str(end_time - start_time) + "seconds)")
        
        """
        pickle.dump(self.first_mover, "./trained_models/first_mover.p")
        pickle.dump(self.second_mover, "./trained_models/second_mover.p")
        """

    def test(self, trained_agent: str):
        winning_record = []
        episode = 0
        while episode < 100:
            # reset the episode
            episode += 1
            board = self.reset()
            winner = 0
            
            # 2 agents keep playing against each other until a winner is decided or there is no more space for placing chess
            while 0 in board and winner == 0:
                in_board = board
                if trained_agent != 'second_mover':
                    board, action = self.first_mover.step(in_board.copy())
                else:
                    board, action = self.first_mover.random_action(in_board.copy())
                winner = self.score(board)
                
                if 0 in board and winner == 0:
                    in_board = board
                    if trained_agent != 'first_mover':
                        board, action = self.second_mover.step(in_board.copy())
                    else:
                        board, action = self.second_mover.random_action(in_board.copy())
                    winner = self.score(board)
                    
            winning_record.append(winner)
            
        print(winning_record.count(1))
        print(winning_record.count(0))
        print(winning_record.count(-1))

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