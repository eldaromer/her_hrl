import tensorflow as tf
from util import NormalizedEnv
from OptionReplayMemory import ReplayMemory
import progressbar
import numpy as np
from gym.wrappers import Monitor

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
	def __init__(self, env, hidden_size=512, num_options=8, max_memory=1000000, tau=0.01, gamma=0.99, batch_size=32, update_frequency=4, freeze_interval=10000):
		self.env = NormalizedEnv(env)
		self.action_space = len(env.action_space.high)
		self.hidden_size = hidden_size
		self.max_memory = max_memory
		self.num_options = num_options
		self.batch_size = batch_size
		self.update_frequency = update_frequency
		self.freeze_interval = freeze_interval
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

	def save(self):
		self.q_omega.save_weights('./OptionCritic/q_omega/manual_save')
		self.term_network.save_weights('./OptionCritic/term/manual_save')
		self.intra_network.save_weights('./OptionCritic/intra/manual_save')

	def record(self):
		recordEnv = Monitor(self.env, "recordings", video_callable=lambda episode_id: True, force="true")
		recordEnv.render(mode="human")
		observation = recordEnv.reset()


		total_reward = 0
		termination = True
		for t in range(10000):

			if termination:
				option = np.argmax(self.q_omega(np.array([observation,])))
				option_onehot = tf.squeeze(tf.one_hot([option], self.num_options, dtype='float64'))
				term_onehot = option_onehot
				option_onehot = tf.reshape(option_onehot, [self.num_options, -1, 1])

			actions = self.intra_network(np.array([observation,]))
			action = tf.squeeze(tf.reduce_sum(actions*option_onehot, 0))
			# Execute action and observe reward/new state
			new_observation, reward, done, info = recordEnv.step(action.numpy())
			recordEnv.render(mode="human")
			total_reward += reward
			if done:
				break
			observation = new_observation
		print('Total reward: {}'.format(total_reward))
		recordEnv.close()

	def train(self, num_ep):
		reward_hist = []
		option_history = []
		frame_count = 0

		widgets = [
			progressbar.Counter(), '/', str(num_ep), ' ',
			' [', progressbar.Timer(), '] ',
			progressbar.Bar(),
			' (', progressbar.ETA(), ') ',
		]

		for ep in progressbar.progressbar(range(num_ep), widgets=widgets):
			observation = self.env.reset()
			option_ratio = [0]*self.num_options
			done = False
			option = np.argmax(self.q_omega(np.array([observation,])))
			option_onehot = tf.squeeze(tf.one_hot([option], self.num_options, dtype='float64'))
			term_onehot = option_onehot
			option_onehot = tf.reshape(option_onehot, [self.num_options, -1, 1])
			termination = False
			while not done:
				frame_count += 1
				if termination:
					option = np.argmax(self.q_omega(np.array([observation,])))
					option_onehot = tf.squeeze(tf.one_hot([option], self.num_options, dtype='float64'))
					term_onehot = option_onehot
					option_onehot = tf.reshape(option_onehot, [self.num_options, -1, 1])

				option_ratio[option] += 1

				actions = self.intra_network(np.array([observation,]))
				action = tf.squeeze(tf.reduce_sum(actions*option_onehot, 0))

				new_observation, reward, done, info = self.env.step(action)

				self.memory.push(observation, action, reward, new_observation, done, option)

				if len(self.memory) > self.batch_size:

					# Actor Update
					with tf.GradientTape() as tape:
						# Q_U estimate
						next_term = tf.gather(tf.gather(self.term_network(np.array([new_observation, ])), 0), option)
						next_q = tf.gather(self.target_q_omega(np.array([new_observation])), 0)
						y = reward + (1-done)*self.gamma*(1-next_term)*tf.gather(next_q, option) + done*self.gamma*next_term*np.argmax(next_q)
						tape.reset()
						actions = self.intra_network(np.array([observation,]))
						action = tf.squeeze(tf.reduce_sum(actions*option_onehot, 0))
						policy_grad = -1 * tf.reduce_sum(action*y)

					policy_gradient = tape.gradient(policy_grad, self.intra_network.trainable_variables)
					self.intra_optimizer.apply_gradients(zip(policy_gradient, self.intra_network.trainable_variables))

					with tf.GradientTape() as tape:
						q = tf.gather(self.target_q_omega(np.array([observation, ])), 0)
						Q = tf.gather(q, option)
						V = tf.reduce_max(q)
						tape.reset()
						term = self.term_network(np.array([observation, ]))
						term = tf.reduce_sum(term*term_onehot)
						term_grad = tf.reduce_sum(term * (Q-V))

					term_gradient = tape.gradient(term_grad, self.term_network.trainable_variables)
					self.term_optimizer.apply_gradients(zip(term_gradient, self.term_network.trainable_variables))

					if frame_count % self.update_frequency == 0:
						state_batch, action_batch, reward_batch, next_state_batch, done_batch, option_batch = self.memory.sample(self.batch_size)

						with tf.GradientTape() as tape:
							next_terms = self.term_network(next_state_batch)
							next_term = [tf.gather(next_terms[i], option_batch[i]) for i in range(len(option_batch))]
							next_q = self.target_q_omega(next_state_batch)
							next_q_option = np.array([tf.gather(next_q[i], option_batch[i]) for i in range(len(option_batch))])
							next_v = tf.reduce_max(next_q, axis=1)
							y_1 = reward + (tf.cast(tf.fill(done_batch.shape, 1.0), tf.float64)-tf.cast(done_batch, tf.float64))*self.gamma*(tf.cast(tf.fill(len(next_term), 1), tf.float64)-next_term)*next_q_option
							y_2 = done_batch*self.gamma*next_term*next_v
							y = y_1 + y_2
							tape.reset()
							option_q = self.q_omega(state_batch)
							option_batch = np.array([tf.squeeze(tf.one_hot([option], self.num_options, dtype=tf.float64)) for option in option_batch])
							option_q = option_q*option_batch
							option_q = tf.reduce_sum(option_q, axis=1)
							td_error = y-option_q
							td_cost = 0.5 * td_error ** 2
							cost = tf.reduce_sum(td_cost)
						cost_gradient = tape.gradient(cost, self.q_omega.trainable_variables)
						self.critic_optimizer.apply_gradients(zip(cost_gradient, self.q_omega.trainable_variables))

					if frame_count % self.freeze_interval == 0:
						self.target_q_omega.set_weights(self.q_omega.get_weights())

				reward_hist.append(reward)
				observation = new_observation
				term = tf.gather(tf.gather(self.term_network(np.array([observation, ])), 0), option)
				if done:
					break
				if np.random.uniform() < term:
					termination = True

			option_history.append(option_ratio)


			print("Episode Done")
		np.save('option_history', option_history)
		np.save('reward_history', reward_hist)


