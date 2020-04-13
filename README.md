# Q-Network
Asynchronous implementation of __Deep Q-Network__ to utilise multi-core + multi-gpu architectures.

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
$ python trainer.py --<arg> <value>
```

##### Command-line arguments

| Argument      | Description                                                   | Values                            | Default       |
|---------------|---------------------------------------------------------------|-----------------------------------|---------------|
| --environment                 | Environment to use for training               | string                            | cartpole      |
| --save_model                  | Path to save the model                        | string                            | ''            |
| --save_model                  | Path to load the model                        | string                            | ''            |
| --n_workers                   | Number of workers to use for training         | int                               | 1             |
| --target_update_frequency     | Sync frequency for target network             | int                               | 10            |
| --checkpoint_frequency        | Frequency for creating checkpoints            | int                               | 10            |
| --lr                          | Learning rate for training                    | float                             | 5e-4          |
| --batch_size                  | Batch size for training                       | int                               | 32            |
| --gamma                       | Discount factor value for training            | float (should be less than 1.0)   | 0.99          |
| --eps                         | Epsilon value for training                    | float (should be less than 1.0)   | 0.999         |
| --min_eps                     | Minimum value for epsilon                     | float                             | 0.1           |
| --buffer_size                 | Buffer size                                   | int                               | 100000        |
| --max_grad_norm               | Maximum L2 norm for gradients                 | float                             | 10            |

# Testing
```shell
$ source .env/bin/activate
$ python tester.py --<arg> <value>
```

##### Command-line arguments

| Argument          | Description                           | Values                        | Default       |
|-------------------|---------------------------------------|-------------------------------|---------------|
| --environment     | Environment to use for testing        | string                        | cartpole      |
| --load_model      | Path to load the model                | Path                          | ''            |

# Custom Simulator
To use a custom simulator, implement the abstract class ```BaseSimulator```.
Implement the QNetwork model by extending ```BaseModel```.
Finally, register the simulator and the model in the ```environments.py```. 

# Reference
V. Mnih et al., “Playing Atari with Deep Reinforcement Learning,” arXiv:1312.5602 [cs], Dec. 2013.
