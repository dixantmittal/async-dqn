from abc import abstractmethod


class ISimulator(object):
    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, a):
        raise NotImplementedError

    @abstractmethod
    def nActions(self):
        raise NotImplementedError

    @abstractmethod
    def nStates(self):
        raise NotImplementedError

    @abstractmethod
    def sampleAction(self):
        raise NotImplementedError
