import gym
from ppo.agent import PPO
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import tensorflow as tf

env_name = 'Acrobot-v1'
env = gym.make(env_name)
critic = Sequential([
    Dense(64, activation='relu'),
    Dense(1, activation='tanh')
])
actor = Sequential([
    Dense(64, activation='relu'),
    Dense(env.action_space.n, activation="tanh")
])

gamma = 1
epsilon = 0.2

mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM) # critic loss function
ppo = PPO(actor=actor, critic=critic, num_actions=env.action_space.n, gamma=gamma, epsilon=epsilon)
ppo.compile(actor_optimizer=Adam(0.003), critic_optimizer=Adam(0.001), critic_loss=mse)

ppo.fit(env, n_max_steps=500, n_max_episodes=1500, verbose=True)
env.close()
ppo.save_weights('ppo_acrobot')

env = gym.make(env_name, render_mode="human")
ppo.test(env, n_episodes=10)
env.close()