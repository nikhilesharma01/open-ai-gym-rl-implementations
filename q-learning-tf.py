import gym
import numpy as np 
import random
import tensorflow as tf 

# Create gym environment
env = gym.make('FrozenLake-v0')

# Reset the tensorflow graph
tf.reset_default_graph()

state_size = env.observation_space.n 
action_size = env.action_space.n 

# Tensorflow connections
inputs = tf.placeholder(shape = [1, 16], dtype = tf.float32)
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))

Q_out = tf.matmul(inputs, W)
predict = tf.argmax(Q_out, 1)

next_Q = tf.placeholder(shape = [1, 4], dtype = tf.float32)
loss = tf.reduce_sum(tf.square(next_Q - Q_out))

# Setect trainer
trainer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

# Train the network
# Initialize learning parameters

gamma = 0.95
e = 0.1

num_episodes = 2000
rewardsList = []
	
with tf.Session() as sess:
	sess.run(init)
	for i in range(num_episodes):
		# Reset environment and get new observation
		s = env.reset()
		rAll = 0
		done = False
		step = 0

		while step < 99:
			step += 1
			a, allQ = sess.run([predict, Q_out], 
				feed_dict = {inputs:np.identity(16)[s:s+1]})

			if np.random.rand(1) < e:
				a[0] = env.action_space.sample()

			# Step through the algorithm
			s_prime, r, done, _= env.step(a[0])

			# Get Q' values by feeding new state through the graph
			Q_prime = sess.run(Q_out, 
				feed_dict = {inputs:np.identity(16)[s_prime:s_prime+1]})

			maxQ_prime = np.max(Q_prime)
			targetQ = allQ

			targetQ[0, a[0]] = r + gamma * maxQ_prime

			# Train network using target and predicted Q values
			_, W1 = sess.run([updateModel, W], 
				feed_dict = {inputs:np.identity(16)[s:s+1], next_Q:targetQ})

			rAll += r
			s = s_prime

			if done == True:
				e = 1./((i/50.) + 10)
				break

		rewardsList.append(rAll)

print("Percent of successful episodes: " + str(sum(rewardsList) / num_episodes) + "%")
