from agents import first_mover, second_mover
from environment import environment
from models import model
import pickle

'''Parameters'''
mode = "train" # "train" means training agent; "player_1" means using computer as first_mover; "player_2" means using computer as second_mover

class runner():
    def __init__(self, env: environment, mode='train'):
        self.env = env
        self.mode = mode

    def start(self):
        if self.mode == 'train':
            self.env.train()
            self.env.test(trained_agent="first_mover")
            self.env.test(trained_agent="second_mover")
            self.env.test(trained_agent="both agents")

        if self.mode == 'player_1':
            self.env.com_as_player_1()

        if self.mode == 'player_2':
            self.env.com_as_player_2()

if __name__ == "__main__":
    first_mover = agent(role=1, model=model)
    second_mover = agent(role=-1, model=model)
    env = environment(first_mover, second_mover)
    runner = runner(env, mode)
    runner.start()