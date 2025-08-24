import gym
import numpy as np
from stable_baselines3 import DQN, PPO

class RLAgent:
    def __init__(self, env_name, algo='DQN'):
        self.env = gym.make(env_name)
        self.algo = algo
        if algo == 'DQN':
            self.model = DQN('MlpPolicy', self.env, verbose=1)
        else:
            self.model = PPO('MlpPolicy', self.env, verbose=1)

    def train(self, timesteps=10000):
        self.model.learn(total_timesteps=timesteps)

    def evaluate(self, episodes=5, max_steps=1000):
        results = []
        for ep in range(episodes):
            obs = self.env.reset()
            ep_reward = 0
            for _ in range(max_steps):
                act, _ = self.model.predict(obs, deterministic=True)
                obs, r, done, _ = self.env.step(act)
                ep_reward += r
                if done:
                    break
            results.append(ep_reward)
        return results
