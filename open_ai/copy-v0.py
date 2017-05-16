import gym
import sys
import os

env = gym.make('Copy-v0')
for i_episode in range(20):
    observation = env.reset()
    t = 0
    while 1:
        t += 1
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(t, env.action_space, observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
