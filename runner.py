import numpy as np
from environment import environment
from agents import agent

'''Parameters'''
mode = 'train' # "train" means training agent; "player_1" means using computer as first_mover; "player_2" means using computer as second_mover
algo = 'DQN' # select an algorithm for training the agent
epsilon = 0.1 # choose the epsilon value
learning_rate = 0.1 # choose the learning rate
total_episode = 100 # Decide total episode
np.random.seed(123)
'''Parameters'''

class runner():
    def __init__(self, env: environment, mode: str, total_episode: int):
        self.env = env
        self.mode = mode
        self.total_episode = total_episode

    def start(self):
        if self.mode == 'train':
            self.env.train(self.total_episode)
            self.env.test(trained_agent="first_mover")
            self.env.test(trained_agent="second_mover")
            self.env.test(trained_agent="both agents")

        if self.mode == 'player_1':
            self.env.com_as_player_1()

        if self.mode == 'player_2':
            self.env.com_as_player_2()

if __name__ == "__main__":
    first_mover = agent(role=1, algo=algo, epsilon=epsilon, learning_rate=learning_rate)
    second_mover = agent(role=-1, algo=algo, epsilon=epsilon, learning_rate=learning_rate)
    env = environment(first_mover, second_mover)
    runner = runner(env, mode, total_episode)
    runner.start()
    
    # Check if 
    for key in runner.env.memory.keys():
        if runner.env.memory[key][-1].sum().sum() not in [0,1]:        
            print(key)