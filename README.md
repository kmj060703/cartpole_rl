# CartPole RL

Deep Reinforcement Learning implementation of the CartPole problem using a parallelized Deep Q-Network (DQN).

## Overview

This project improves training efficiency by running multiple environments in parallel and provides real-time visualization for monitoring the learning process.

## Features

* Parallel environment execution (9 instances)
* Deep Q-Network (DQN)

  * Experience Replay
  * Target Network
* Real-time visualization using OpenCV

## Installation

```bash
git clone https://github.com/kmj060703/cartpole_rl.git
cd cartpole_rl
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python train.py
```

## Future Work

* Model checkpoint saving/loading
* Hyperparameter tuning
* Performance optimization

## License

MIT License
