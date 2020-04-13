import random

import torch as t

from device import Device


def copy_gradients(target, source):
    for shared_param, param in zip(target.parameters(), source.parameters()):
        if param.grad is not None:
            shared_param._grad = param.grad.clone().cpu()


def prepare_batch(buffer, batch_size):
    batch_size = min(len(buffer), batch_size)
    batch = random.sample(buffer, batch_size)

    states, actions, rewards, next_states, terminal = zip(*batch)

    return t.stack(states), t.stack(actions), t.stack(rewards), t.stack(next_states), t.stack(terminal)


def as_tensor(x, dtype=t.float32):
    return t.tensor(x, dtype=dtype, device=Device.get_device())


def transition_to_tensor(state, action, reward, next_state, terminal):
    return (as_tensor(state),
            as_tensor([action], t.long),
            as_tensor([reward]),
            as_tensor(next_state),
            as_tensor([not terminal]))
