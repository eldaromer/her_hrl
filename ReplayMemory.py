import numpy as np
from collections import deque
import random

class ReplayMemory:
	def __init__(self, size):
		self.buffer = deque(maxlen=size)
		
	def push(self, state, action, reward, next_state, done):
		new_experience = (state, action, [reward], next_state, done)
		self.buffer.append(new_experience)
		
	def sample(self, batch_size):
		state_batch = []
		action_batch = []
		reward_batch = []
		next_state_batch = []
		done_batch = []

		batch = random.sample(self.buffer, batch_size)

		for experience in batch:
			state, action, reward, next_state, done = experience
			state_batch.append(state)
			action_batch.append(action)
			reward_batch.append(reward)
			next_state_batch.append(next_state)
			done_batch.append(done)
		
		return np.array(state_batch), np.array(action_batch), np.array(reward_batch), np.array(next_state_batch), np.array(done_batch)
		
	def __len__(self):
		return len(self.buffer)