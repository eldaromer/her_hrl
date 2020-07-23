import numpy as np
from collections import deque
import random

class ReplayMemory:
	def __init__(self, size, env, k=4):
		self.buffer = deque(maxlen=size)
		self.temp = []
		self.env = env
		self.k = k

	def push(self, state, action, reward, next_state, done, goal, info):
		new_experience = (state, action, [reward], next_state, done, goal, info)
		self.temp.append(new_experience)

	def back(self):
		sampled = random.sample(self.temp, self.k)
		sampled_goals = [sample[5] for sample in sampled]
		for experience in self.temp:
			self.buffer.push(experience)
			for goal in sampled_goals:
				reward = env.compute_reward(experience[5], goal, experience[6])
				new_experience = (experience[0], experience[1], [reward], experience[3], experience[4], goal, experience[6])
				self.buffer.push(new_experience)
		self.temp = []


	def sample(self, batch_size):
		state_batch = []
		action_batch = []
		reward_batch = []
		next_state_batch = []
		done_batch = []

		batch = random.sample(self.buffer, batch_size)

		for experience in batch:
			state, action, reward, next_state, done, goal, info = experience
			state_batch.append(state)
			action_batch.append(action)
			reward_batch.append(reward)
			next_state_batch.append(next_state)
			done_batch.append(done)
		
		return np.array(state_batch), np.array(action_batch), np.array(reward_batch), np.array(next_state_batch), np.array(done_batch)
		
	def __len__(self):
		return len(self.buffer)