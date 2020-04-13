import torch as t
import torch.nn.functional as f


def calculate_loss(q_network, target_network, batch, hyperparameters):
    state, action, reward, next_state, terminal = batch

    with t.no_grad():
        target = reward + terminal * hyperparameters.gamma * target_network(next_state).max()

    predicted = q_network(state).gather(1, action)

    return f.smooth_l1_loss(predicted, target)
