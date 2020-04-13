from abc import abstractmethod


class BaseSimulator(object):
    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, a):
        raise NotImplementedError

    @staticmethod
    def n_actions():
        raise NotImplementedError

    @staticmethod
    def state_shape():
        raise NotImplementedError
