import numpy as np
import tensorflow as tf
import gym
from typing import Tuple
import collections
import tqdm
import statistics
import keras
import os
from keras.callbacks import CallbackList

class PPO(keras.Model):
    def __init__(self, actor: tf.keras.Model, critic: tf.keras.Model, num_actions, gamma: float = 1, epsilon: float = 0.2):
        super().__init__()
        
        if epsilon >= 1 or epsilon <= 0:
            raise ValueError('`epsilon` must be > 0 and < 1')

        self.critic = critic
        self.actor = actor
        self.num_actions = num_actions

        self.gamma = gamma
        self.epsilon = epsilon

        self.compiled = False

    def compile(self, actor_optimizer, critic_optimizer, critic_loss):
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.critic_loss = critic_loss

        self.actor.compile(actor_optimizer)
        self.critic.compile(critic_optimizer)
        
        self.compiled = True

    def call(self, inputs, training=False):
        return self.actor(inputs), self.critic(inputs)

    def test(self, env: gym.Env, n_episodes: int = 1, n_max_steps: int = 500):
        for _ in range(n_episodes):
            observation, _ = env.reset()
            for _ in range(n_max_steps):
                action_logits, value = self(tf.expand_dims(observation, 0))
                action = tf.random.categorical(action_logits, 1)[0, 0].numpy()
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

    def fit(self, env: gym.Env, n_max_episodes: int = 500, n_max_steps: int = 500, mean_queue_size=100, verbose=False, _episode_callback=None):
        if self.compiled == False:
            raise RuntimeError('Run `compile()` before fitting this model.')

        self._t_env = env
        t = tqdm.trange(n_max_episodes, disable=not verbose)
        episodes_reward = collections.deque(maxlen=mean_queue_size)

        for _ in t:
            inital_observation, _ = env.reset()
            inital_observation = tf.constant(inital_observation, dtype=tf.float32)

            reward = int(self._train_iteration(inital_observation, n_max_steps))
            episodes_reward.append(reward)
            
            mean_reward=statistics.mean(episodes_reward)
            t.set_postfix(rewards=reward, mean_reward=mean_reward)
            if _episode_callback is not None:
                _episode_callback(reward, mean_reward)

        self._t_env = None
    
    def _log_probability(self, action_logits_t, action):
        all_logits = tf.nn.log_softmax(action_logits_t)
        log_probabilities = tf.reduce_sum(tf.one_hot(action, self.num_actions) * all_logits, axis=1)
        return log_probabilities


    @tf.function
    def _train_iteration(self, inital_state, n_max_steps: int):
        states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        actions = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)

        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        state = inital_state
        for step in tf.range(n_max_steps):
            state = tf.expand_dims(state, 0)
            states = states.write(step, state)

            action_logits_t = self.actor(state)
            value = self.critic(state)

            action = tf.squeeze(tf.random.categorical(action_logits_t, 1))
            actions = actions.write(step, action)

            values = values.write(step, tf.squeeze(value))
            action_probs = action_probs.write(step, self._log_probability(action_logits_t, action))

            state, reward, terminated = tf.numpy_function(self._run_single_step, [action], [tf.float32, tf.int32, tf.int32])
            state.set_shape(inital_state.shape)

            rewards = rewards.write(step, reward)
            if tf.cast(terminated, tf.bool):        
                break

        states = states.stack()
        actions = actions.stack()
        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()    

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + self.gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]
        eps = np.finfo(np.float32).eps.item() # Stabillity constant in case on /0
        returns = ((returns - tf.math.reduce_mean(returns)) / 
                (tf.math.reduce_std(returns) + eps))

        with tf.GradientTape() as actor_tape:
            old_action_logs = tf.squeeze(self.actor(states))
            old_action_probs = self._log_probability(old_action_logs, actions)  
            advantage = returns - values
            old_action_probs = tf.squeeze(old_action_probs)
            action_probs = tf.squeeze(action_probs)

            prob_ratio = tf.exp(old_action_probs - action_probs)       
            Lcpi = prob_ratio * advantage            
            Lcpiclipped = tf.clip_by_value(prob_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
            Lclip = -tf.math.reduce_mean(tf.minimum(Lcpi, Lcpiclipped))
            
        grads = actor_tape.gradient(Lclip, self.actor.trainable_variables)                  
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        with tf.GradientTape() as critic_tape:
            critic_loss =  self.critic_loss(self.critic(states), returns)
        
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        return tf.math.reduce_sum(rewards)

    def _run_single_step(self, action):
        observation, reward, terminated, _, _ = self._t_env.step(action)
        return (
            observation.astype(np.float32),
            np.array(reward, np.int32),
            np.array(terminated, np.int32)
        )

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)
