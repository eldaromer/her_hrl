import tensorflow as tf
from util import NormalizedEnv
from ReplayMemory import ReplayMemory
import progressbar
import numpy as np

class TerminationOption(tf.keras.layers.Layer):
	def __init__(self, hidden_size):
		super(TerminationOption, self).__init__()
		self.Dense1 = tf.keras.layers.Dense(hidden_size, activation="relu")
		self.out = tf.keras.layers.Dense(1, activation="sigmoid")

	def call(self, x):
		x = self.Dense1(x)
		return self.out(x)

class TerminationNetwork(tf.keras.Model):
	def __init__(self, hidden_size, num_options):
		super(TerminationNetwork, self).__init__()
		self.termination_layers = [TerminationOption(hidden_size) for _ in range(num_options)]

	def call(self, inputs):
		term_probs = [term(inputs) for term in self.termination_layers]
		return tf.concat(term_probs, axis=1)

class IntraOption(tf.keras.layers.Layer):
	def __init__(self, hidden_size, output_size):
		super(IntraOption, self).__init__()
		self.Dense1 = tf.keras.layers.Dense(hidden_size, activation="relu")
		self.out = tf.keras.layers.Dense(output_size, activation="tanh")

	def call(self, inputs):
		x = self.Dense1(inputs)
		return self.out(x)

class IntraOptionNetwork(tf.keras.Model):
	def __init__(self, hidden_size, output_size, num_options):
		super(IntraOptionNetwork, self).__init__()
		self.intra_layers = [IntraOption(hidden_size, output_size) for _ in range(num_options)]

	def call(self, inputs):
		action_probs = [intra(inputs) for intra in self.intra_layers]
		return tf.convert_to_tensor(action_probs)


class Q_Omega(tf.keras.Model):
	def __init__(self, hidden_size, output_size):
		super(Q_Omega, self).__init__()
		self.Dense1 = tf.keras.layers.Dense(hidden_size, activation="relu")
		self.out = tf.keras.layers.Dense(output_size)

	def call(self, inputs):
		x = self.Dense1(inputs)
		return self.out(x)

class OptionCritic:
	def __init__(self, env, hidden_size=512, num_options=8, max_memory=50000, tau=0.01, gamma=0.99, batch_size=256):
		self.env = NormalizedEnv(env)
		self.action_space = len(env.action_space.high)
		self.hidden_size = hidden_size
		self.max_memory = max_memory
		self.batch_size = batch_size
		self.gamma = gamma
		self.tau = tau
		self.memory = ReplayMemory(max_memory)
		self.term_network = TerminationNetwork(hidden_size, num_options)
		self.intra_network = IntraOptionNetwork(hidden_size, self.action_space, num_options)
		self.q_omega = Q_Omega(hidden_size, num_options)
		self.target_q_omega = Q_Omega(hidden_size, num_options)
		self.target_q_omega.set_weights(self.q_omega.get_weights())
		self.critic_optimizer = tf.keras.optimizers.Adam(lr=0.001)
		self.intra_optimizer = tf.keras.optimizers.Adam(lr=0.0001)
		self.term_optimizer = tf.keras.optimizers.Adam(lr=0.0001)
	@tf.function
	def train(self, num_ep):
		reward_hist = []

		widgets = [
			progressbar.Counter(), '/', str(num_ep), ' ',
			' [', progressbar.Timer(), '] ',
			progressbar.Bar(),
			' (', progressbar.ETA(), ') ',
		]

		for ep in progressbar.progressbar(range(num_ep), widgets=widgets):
			observation = self.env.reset()
			done = False
			option = tf.math.argmax(self.q_omega(np.array([observation,])))
			termination = False
			while not done:
				if termination:
					option = tf.math.argmax(self.q_omega(np.array([observation,])))

				actions = self.intra_network(np.array([observation,]))
				action = tf.gather(tf.gather(actions, option), 0)

				new_observation, reward, done, info = self.env.step(action)

				self.memory.push(observation, action, reward, new_observation, done)

				if len(self.memory) > self.batch_size:

					# Actor Update
					with tf.GradientTape() as tape:
						# Q_U estimate
						tape.watch(action)
						next_term = tf.gather(tf.gather(self.term_network(np.array([new_observation, ])), 0), option)
						next_q = tf.gather(self.target_q_omega(np.array([new_observation])), 0)
						y = reward + (1-done)*self.gamma*(1-next_term)*tf.gather(next_q, option) + done*self.gamma*next_term*np.argmax(next_q)
						tape.reset()
						policy_grad = -1 * tf.reduce_sum(np.log(action)*y)

					policy_gradient = tape.gradient(policy_grad, self.intra_network.trainable_variables)
					print(policy_gradient)
					self.intra_optimizer.apply_gradients(zip(policy_gradient, self.intra_network.trainable_variables))

					with tf.GradientTape() as tape:
						q = tf.gather(self.target_q_omega(np.array([observation, ])), 0)
						Q = tf.gather(q, option)
						V = tf.reduce_max(q)
						tape.reset()
						term = tf.gather(tf.gather(self.term_network(np.array([observation, ])), 0), option)
						term_grad = tf.reduce_sum(term * (Q-V))

					term_gradient = tape.gradient(term_grad, self.term_network.trainable_variables)
					print(term_gradient)
					self.term_optimizer.apply_gradients(zip(term_gradient, self.term_network.trainable_variables))

					#state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)

					#with tf.GradientTape() as tape:


			print("Episode Done")
			#print(actions)


