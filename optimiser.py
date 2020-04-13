import logging
from collections import deque
from copy import deepcopy
from datetime import datetime
from itertools import count

import numpy as np
import torch as t
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from calculate_loss import calculate_loss
from device import Device
from optimise_model import optimise_model
from utils import prepare_batch, as_tensor, transition_to_tensor


def optimiser(idx, shared_model, SIMULATOR, args, lock):
    try:
        writer = SummaryWriter('runs/{}/optimiser:{:02}'.format(datetime.now().strftime("%d|%m_%H|%M"), idx))
        logging.basicConfig(filename='logs/optimiser:{:02}.log'.format(idx),
                            filemode='w',
                            format='%(message)s',
                            level=logging.DEBUG)

        sgd = t.optim.SGD(params=shared_model.parameters(), lr=args.lr)

        # allocate a device
        n_gpu = t.cuda.device_count()
        if n_gpu > 0:
            Device.set_device(idx % n_gpu)

        q_network = deepcopy(shared_model)
        q_network.to(Device.get_device())
        q_network.train()

        target_network = deepcopy(q_network)
        target_network.to(Device.get_device())
        target_network.eval()

        buffer = deque(maxlen=args.buffer_size)

        simulator = SIMULATOR()
        for itr in tqdm(count(), position=idx, desc='optimiser:{:02}'.format(idx)):

            state = simulator.reset()
            episode_reward = 0
            for e in count():
                if np.random.RandomState().rand() < max(args.eps ** itr, args.min_eps):
                    action = np.random.RandomState().randint(simulator.n_actions())
                else:
                    action = q_network(as_tensor(state)).argmax().item()

                next_state, reward, terminal = simulator.step(action)

                buffer.append(transition_to_tensor(state, action, reward, next_state, terminal))

                episode_reward += reward
                state = next_state

                # Sample a data point from dataset
                batch = prepare_batch(buffer, args.batch_size)

                # Sync local model with shared model
                q_network.load_state_dict(shared_model.state_dict())

                # Calculate loss for the batch
                loss = calculate_loss(q_network, target_network, batch, args)

                # Optimise for the batch
                loss = optimise_model(shared_model, q_network, loss, sgd, args, lock)

                # Log the results
                logging.debug('Batch loss: {:.2f}'.format(loss))
                writer.add_scalar('batch/loss', loss, e)

                if terminal:
                    break

            logging.debug('Episode reward: {:.2f}'.format(episode_reward))
            writer.add_scalar('episode_reward', episode_reward, itr)
            writer.close()

            if itr % args.target_update_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    except KeyboardInterrupt:
        print('exiting optimiser:{:02}'.format(idx))
