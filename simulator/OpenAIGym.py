import gym
import numpy

from simulator.ISimulator import ISimulator


class CartPole(ISimulator):
    def __init__(self):
        self.env = gym.make('CartPole-v1')

    def reset(self):
        return self.env.reset()

    def step(self, a):
        return self.env.step(a)

    def nActions(self):
        return self.env.action_space.n

    def dState(self):
        return self.env.observation_space.shape[0]

    def sampleAction(self):
        return numpy.random.randint(self.nActions())

    def prettifyState(self, rawState):
        return rawState.reshape(-1, self.dState())

    def __del__(self):
        self.env.close()
