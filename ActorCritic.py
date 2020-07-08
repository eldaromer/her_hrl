import tensorflow as tf

class Actor(tf.keras.Model):
	def __init__(self, hidden_size, output_size):
		super(Actor, self).__init__()
		self.Dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
		self.Dense2 = tf.keras.layers.Dense(hidden_size, activation='relu')
		self.out = tf.keras.layers.Dense(output_size, activation='tanh')
		
	def call(self, state):
		x = self.Dense1(state)
		x = self.Dense2(x)
		return self.out(x)
	
class Critic(tf.keras.Model):
	def __init__(self, hidden_size, output_size):
		super(Critic, self).__init__()
		self.Dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
		self.Dense2 = tf.keras.layers.Dense(hidden_size, activation='relu')
		self.out = tf.keras.layers.Dense(output_size)
		
	def call(self, state, actions):
		x = tf.concat([state, actions], 1)
		x = self.Dense1(x)
		x = self.Dense2(x)
		return self.out(x)