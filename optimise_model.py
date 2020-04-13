from torch.nn.utils import clip_grad_norm_

from utils import copy_gradients


def optimise_model(shared_model, local_model, loss, optimiser, args, lock):
    # Compute gradients
    loss.backward()

    # Optimise by taking a gradient step
    clip_grad_norm_(local_model.parameters(), args.max_grad_norm)

    # The critical section begins
    lock.acquire()
    copy_gradients(shared_model, local_model)
    optimiser.step()
    lock.release()
    # The critical section ends

    local_model.zero_grad()

    return loss.item()
