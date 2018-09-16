import numpy as np
from environment import environment
from agents import agent

'''Parameters'''
# Runner Parameters
mode = 'train' # "train" means training agent; "player_1" means using computer as first_mover; "player_2" means using computer as second_mover
total_episode = 1000 # Decide total episode
np.random.seed(123)

# Model Parameters
batch_size = 32
learning_rate = 0.05
epsilon = 0.1
gamma = 0.9
target_replace_iter = 100
memory_capacity = 1000
n_actions = 7
n_states = 42

'''Runner'''
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
    first_mover = agent(1, epsilon, learning_rate, gamma, batch_size, target_replace_iter, memory_capacity, n_actions, n_states)
    second_mover = agent(-1, epsilon, learning_rate, gamma, batch_size, target_replace_iter, memory_capacity, n_actions, n_states)
    env = environment(first_mover, second_mover)
    runner = runner(env, mode, total_episode)
    runner.start()