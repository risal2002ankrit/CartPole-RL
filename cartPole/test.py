# import torch as T
# device = T.device("cuda:0" if T.cuda.is_available() else 'cpu')
# print(device)

import gym
env = gym.make('CartPole-v1')
print(env.observation_space.shape)

