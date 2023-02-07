# Proximal Policy Iteration

This repository contains an actor-critic implementation of a proximal policy iteration

## Installation

Simply pull this repository and execute `pip install -e .` in a command line.

## Usage

Working examples can be found in the `examples/` folder

## Note

This implemntation assumes you provide your own actor network, critic network and optimizers:

```python
from ppo.agent import PPO
from keras.models import Sequential
from keras.layers import Dense

critic = Sequential([
    Dense(64, activation='relu'),
    Dense(1, activation='tanh')
])
actor = Sequential([
    Dense(64, activation='relu'),
    Dense(..., activation="tanh")
])
ppo = PPO(actor=actor, critic=critic, num_actions=..., gamma=..., epsilon=...)
ppo.compile(actor_optimizer=..., critic_optimizer=..., critic_loss=...)
ppo.fit(...)
```

## Requirements

```
tensorflow
keras
gym
```
