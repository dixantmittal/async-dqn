# Q-Network
Implementation of __Deep Q-Network__ for sequential decision-making problems such as atari games, cart pole, mountain car. 

# Requirements
* Python 3

# Installation
```shell
$ python3 -m venv .env
$ source .env/bin/activate
$ pip install torch torchvision gym
```

# Training
```shell
$ source .env/bin/activate
$ python Train.py --<arg> <value>
```

##### Command-line arguments

| Argument      | Description                                                   | Values                          | Default         |
|---------------|---------------------------------------------------------------|---------------------------------|-----------------|
| --simulator   | Name of the Simulator class to use for training               | ClassName                       | None (required) |
| --network     | Path to save the trained network                              | Path                            | None (required) |
| --lr          | Learning rate for training                                    | float                           | 1e-4            |
| --eps         | Epsilon value for training                                    | float                           | 0.999           |
| --batch       | Batch size for training                                       | int                             | 32              |
| --itr         | Number of iterations for training [0 for infinite]            | int (0 for infinite)            | 0               |
| --threads     | Number of parallel simulator instances to collect experiences | int                             | 2               |
| --gamma       | Gamma value for training                                      | float (should be less than 1.0) | 0.99            |
| --frequency   | Target network update frequency                               | int                             | 50              |
| --memory      | Buffer size (in number of experiences)                        | int                             | 100000          |
| --logger      | Logging sensitivity                                           | [debug, info, warn]             | info            |
| --test_size   | Size of the test set                                          | int                             | 100             |
| --device      | Device to use for training                                    | [cpu, cuda]                     | cpu             |
| --checkpoints | store checkpoints while training                              | bool                            | False           |

# Testing
```shell
$ source .env/bin/activate
$ python Test.py --<arg> <value>
```

##### Command-line arguments

| Argument      | Description                                                   | Values                          | Default         |
|---------------|---------------------------------------------------------------|---------------------------------|-----------------|
| --simulator   | Name of the Simulator class to use for testing                | ClassName                       | None (required) |
| --network     | Path to the trained network                                   | Path                            | None (required) |
| --logger      | Logging sensitivity                                           | [debug, info, warn]             | info            |
| --device      | Device to use for testing                                     | [cpu, cuda]                     | cpu             |

# Custom Simulator
To use custom simulator, implement the abstract class ```ISimulator.py``` and import it in the ```simulator/__init__.py```.
```QNetwork.py``` may also need to be modified depending on the input and choice of architecture design. Finally, run the training by passing the new simulator class name as an argument.

# Reference
V. Mnih et al., “Playing Atari with Deep Reinforcement Learning,” arXiv:1312.5602 [cs], Dec. 2013.
