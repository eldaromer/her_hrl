from ActorCritic import Actor, Critic
from ReplayMemory import ReplayMemory
from util import OUNoise, NormalizedEnv
import progressbar
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
import numpy as np
from gym.wrappers import Monitor

tf.keras.backend.set_floatx('float64')

class DDPG:
	def __init__(self, env, batch_size=256, max_memory=500000, tau=0.01, gamma=0.99, hidden_size=400, verbose=True):
		self.env = NormalizedEnv(env)
		self.batch_size = batch_size
		self.max_memory = max_memory
		self.tau = tau
		self.gamma = gamma
		self.hidden_size = hidden_size
		self.verbose = verbose
		self.noise = OUNoise(env.action_space)
		self.memory = ReplayMemory(max_memory)

		self.action_space = len(env.action_space.high)

		self.actor = Actor(hidden_size, self.action_space)
		self.critic = Critic(hidden_size, self.action_space)

		self.actor_target = Actor(hidden_size, self.action_space)
		self.critic_target = Critic(hidden_size, self.action_space)

		self.actor_target.set_weights(self.actor.get_weights())
		self.critic_target.set_weights(self.critic.get_weights())

		self.critic_optimizer = tf.keras.optimizers.Adam(lr=0.001)
		self.actor_optimizer = tf.keras.optimizers.Adam(lr=0.0001)

	def train(self, num_ep):

		reward_hist = []

		widgets = [
							 progressbar.Counter(), '/', str(num_ep), ' ',
							 ' [', progressbar.Timer(), '] ',
								progressbar.Bar(),
							' (', progressbar.ETA(), ') ',
		]

		for ep in progressbar.progressbar(range(num_ep), widgets=widgets):
			self.noise.reset()
		
			# Get Initial Observation
			observation = self.env.reset()
			
			# Reset Stats
			total_reward = 0

			for t in range(10000):
				# Select action from policy and noise
				action = self.actor(np.array([observation,]))[0]
				action = self.noise.get_action(action, t)
				
				# Execute action and observe reward/new state
				new_observation, reward, done, info = self.env.step(action)
				
				# Store transition is replay memory
				self.memory.push(observation, action, reward, new_observation, done)
				
				# Sample minibatch and perform update
				if len(self.memory) > self.batch_size:
					state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
					
					# Update Critic
					with tf.GradientTape() as tape:
							next_act = self.actor_target(next_state_batch)
							next_q = self.critic_target(next_state_batch, next_act)
							y = reward_batch + self.gamma * next_q
							cur_q = self.critic(state_batch, action_batch)
							loss = tf.reduce_mean((y-cur_q)**2)
					grad = tape.gradient(loss, self.critic.trainable_variables)
					self.critic_optimizer.apply_gradients(zip(grad, self.critic.trainable_variables))
					
					# Update Actor
					with tf.GradientTape() as tape:
							new_q = self.critic(state_batch, self.actor(state_batch))
							loss = -1 * tf.reduce_mean(new_q)
					grad = tape.gradient(loss, self.actor.trainable_variables)
					self.actor_optimizer.apply_gradients(zip(grad, self.actor.trainable_variables))
					
					# Update target networks
					self.actor_target.set_weights(self.tau * np.array(self.actor.get_weights()) + (1 - self.tau) * np.array(self.actor_target.get_weights()))
					self.critic_target.set_weights(self.tau * np.array(self.critic.get_weights()) + (1 - self.tau) * np.array(self.critic_target.get_weights()))
						
				
				total_reward += reward
				if done:
						break
				observation = new_observation
				
			if self.verbose:
				print("Episode {} completed: {}".format(ep+1, total_reward))
			reward_hist.append(total_reward)

		self.env.close()
		return reward_hist

	def test(self, num_eps):
		observation = self.env.reset()
		reward_hist = []

		for ep in range(num_eps):
				total_reward = 0
				observation = self.env.reset()
				for t in range(10000):
						# Select action from policy and noise
						action = self.actor(np.array([observation,]))[0]

						# Execute action and observe reward/new state
						new_observation, reward, done, info = self.env.step(action.numpy())
						total_reward += reward
						if done:
								break
						observation = new_observation
				reward_hist.append(total_reward)
		self.env.close()

		return reward_hist

	def record(self):
		recordEnv = Monitor(self.env, "recordings", video_callable=lambda episode_id: True, force="true")
		recordEnv.render(mode="human")
		observation = recordEnv.reset()


		total_reward = 0
		for t in range(10000):
				# Select action from policy and noise
				action = self.actor(np.array([observation,]))[0]
				# Execute action and observe reward/new state
				new_observation, reward, done, info = recordEnv.step(action.numpy())
				recordEnv.render(mode="human")
				total_reward += reward
				if done:
						break
				observation = new_observation
		print('Total reward: {}'.format(total_reward))
		recordEnv.close()

	def save(self):
		self.actor.save_weights('./actor/manual_save')
		self.critic.save_weights('./critic/manual_save')