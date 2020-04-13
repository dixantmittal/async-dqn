import gym

from simulator.base import BaseSimulator


class CartPole(BaseSimulator):
    def __init__(self):
        self.env = gym.make('CartPole-v1')

    def reset(self):
        return self.env.reset()

    def step(self, a):
        s, r, t, _ = self.env.step(a)
        return s, r, t

    @staticmethod
    def n_actions():
        return gym.make('CartPole-v1').action_space.n

    @staticmethod
    def state_shape():
        return gym.make('CartPole-v1').observation_space.shape[0]

    def __del__(self):
        self.env.close()
