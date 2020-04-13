from copy import deepcopy

import torch as t

from device import Device
from utils import as_tensor


def performer(idx, model, SIMULATOR):
    # allocate a device
    n_gpu = t.cuda.device_count()
    if n_gpu > 0:
        Device.set_device(idx % n_gpu)

    q_network = deepcopy(model)
    q_network.to(Device.get_device())
    q_network.eval()

    simulator = SIMULATOR()

    state = simulator.reset()
    episode_reward = 0

    terminal = False
    while not terminal:
        action = q_network(as_tensor(state)).argmax().item()

        next_state, reward, terminal = simulator.step(action)

        episode_reward += reward
        state = next_state

    return episode_reward
