import argparse

import torch as t

from environments import environments
from performer import performer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', default='cartpole', help='Environment to use for training [default = cartpole]')
    parser.add_argument('--load_model', default='', help='Path to load the model [default = '']')
    args = parser.parse_args()

    SIMULATOR, NETWORK = environments[args.environment]
    model = NETWORK()
    model.load(args.load_model)

    with t.no_grad():
        performer(0, model, SIMULATOR)
