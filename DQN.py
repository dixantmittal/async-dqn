import threading
import time

import numpy
import torch

import Logger
from ReplayMemory import ReplayMemory
from SimulatorFactory import SimulatorFactory

logger = Logger.logger


class ExperienceCollector(object):
    def __init__(self, id, network, args):
        self.network = network
        self.args = args
        self.name = 'ExperienceCollector_{}'.format(id)

        self.stopThread = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.collect, name=self.name)
        self.thread.start()
        logger.info('Started thread: %s', self.name)

    def collect(self):
        # Parameters
        eps = self.args.eps
        gamma = self.args.gamma
        device = self.args.device

        simulator = SimulatorFactory.getInstance(self.args.simulator)
        buffer = ReplayMemory(self.args.memory)

        itr = 0
        while not self.stopThread:
            eps = max(0.1, eps ** itr)

            done = False
            episodeReward = 0
            episodeLength = 0

            self.lock.acquire()
            try:
                policyNetwork = self.network.to(device)

                # Reset simulator for new episode
                logger.debug('Starting new episode')

                state = simulator.reset()
                while not done and not DQN.stopExperienceCollectors:
                    action = simulator.sampleAction()
                    if numpy.random.rand() > eps:
                        action = policyNetwork(torch.Tensor(state).to(device)).argmax().item()

                    # take action and get next state
                    nextState, reward, done, _ = simulator.step(action)

                    # store into experience memory
                    buffer.push(state, action, nextState, reward, int(not done))

                    state = nextState
                    episodeReward += reward * gamma ** episodeLength
                    episodeLength += 1

                    if gamma ** episodeLength < 0.1:
                        break
            finally:
                self.lock.release()

        buffer.stop()
        logger.info('Stopped thread: %s', self.name)

    def stop(self):
        self.stopThread = True
        self.thread.join()


class DQN(object):
    experienceCollectors = []
    stopExperienceCollectors = False
    threadSync = None
    stopSyncThread = False

    network = None

    @staticmethod
    def to_one_hot(indices, n_classes):
        one_hot = torch.zeros(len(indices), n_classes)
        one_hot[torch.arange(len(indices)), indices.to(torch.long).squeeze()] = 1
        return one_hot

    @staticmethod
    def syncNetwork():
        while not DQN.stopSyncThread:
            for experienceCollector in DQN.experienceCollectors:
                logger.info('Syncing policy network with %s', experienceCollector.name)
                experienceCollector.lock.acquire()
                try:
                    experienceCollector.network = DQN.network.copy()
                finally:
                    experienceCollector.lock.release()
            time.sleep(2)

        logger.info('Stopped thread: Network Sync')

    @staticmethod
    def train(network, args):
        DQN.network = network

        # Parameters
        gamma = args.gamma
        device = args.device

        simulator = SimulatorFactory.getInstance(args.simulator)
        nStates = simulator.nStates()
        nActions = simulator.nActions()

        # Initialise Metrics
        metrics = {
            'test_set': [],
            'best_test_performance': -numpy.inf
        }

        DQN.experienceCollectors = [ExperienceCollector(i, network, args) for i in range(args.threads)]
        DQN.threadSync = threading.Thread(target=DQN.syncNetwork, name='SyncThread')
        DQN.threadSync.start()
        logger.info('Started thread: Network Sync')

        while ReplayMemory.memoryEmpty:
            time.sleep(0.1)

        # initialise a test set
        logger.debug('Loading test set')
        test = []
        for i in range(args.testSize):
            test.append(simulator.reset())
        logger.debug('Test set loaded!')
        test = torch.Tensor(test).to(device)

        policyNetwork = network.to(device)
        targetNetwork = policyNetwork.copy().to(device)

        # initialise optimiser and loss function
        optimiser = torch.optim.Adam(policyNetwork.parameters(), lr=args.lr, weight_decay=1e-4)
        lossFn = torch.nn.MSELoss()

        itr = 0
        while args.itr == 0 or itr < args.itr:
            itr += 1

            if itr % args.frequency == 0:
                targetNetwork = policyNetwork.copy().to(device)

            # OPTIMIZE POLICY
            batch = ReplayMemory.sample(args.batchSize)

            # slice them to get state and actions
            batch = torch.Tensor(batch).to(device)
            state, action, next_state, reward, terminate = torch.split(batch, [nStates, 1, nStates, 1, 1], dim=1)

            action = DQN.to_one_hot(action, nActions).to(device)

            # find the target value
            target = reward + terminate * gamma * targetNetwork(next_state).max(dim=1)[0].unsqueeze(dim=1)

            # Calculate Q value
            predicted = (policyNetwork(state) * action).sum(dim=1).unsqueeze(dim=1)

            # find loss
            loss = lossFn(predicted, target)

            # Backprop
            optimiser.zero_grad()
            loss.backward()

            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad.clip_grad_value_(policyNetwork.parameters(), 5)
            optimiser.step()

            # Terminate episode if contribution of next state is small.
            # Store Evaluation Metrics
            metrics['test_set'].append(policyNetwork(test).max(dim=1)[0].mean().item())

            # Print statistics
            logger.info('[Iteration: %s] Test Q-values: %s', itr, metrics['test_set'][-1])

            # Checkpoints
            if metrics['test_set'][-1] > metrics['best_test_performance']:
                metrics['best_test_performance'] = metrics['test_set'][-1]

                policyNetwork.save(args.networkPath)
                if args.checkpoints:
                    policyNetwork.save('checkpoints/Q_network_{}.pth'.format(metrics['test_set'][-1]))

    @staticmethod
    def stop():
        DQN.stopExperienceCollectors = True
        for collector in DQN.experienceCollectors:
            collector.stop()

        DQN.stopSyncThread = True
        if DQN.threadSync is not None:
            DQN.threadSync.join()
