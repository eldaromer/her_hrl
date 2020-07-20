import gym
import pybullet_envs
import matplotlib.pyplot as plt
import numpy as np
from DDPG import DDPG
from OptionCritic import OptionCritic

env = gym.make('HalfCheetahBulletEnv-v0')

agent = OptionCritic(env)
agent.train(200)
agent.save()
agent.record()

# agent = DDPG(env, verbose=False)
# train_hist = agent.train(5)
# test_hist = agent.test(5)
# agent.record()
# agent.save()

# fig = plt.figure(figsize=(10, 4))
# ax = fig.add_subplot(121)
# ax.set_xlabel('Episode')
# ax.set_ylabel('Reward')
# ax.set_title('Training History')

# ax.plot(list(range(1, len(train_hist)+1)), train_hist)

# ax2 = fig.add_subplot(122)
# ax2.set_xlabel('Episode')
# ax2.set_ylabel('Reward')
# ax2.set_title('Test Results')

# ax2.plot(list(range(1, len(test_hist)+1)), test_hist)

# plt.show()