import tensorflow as tf 
import numpy as np 

# Initialize bandits
bandits = [0.2, 0, -0.2, 5]
num_bandits = len(bandits)

def pullBandit(bandit):
	# Get a random number
	result = np.random.randn(1)

	if result > bandit:
		# Return a positive reward
		return 1
	else:
		# Return a negative reward
		return -1

# Establish the agent
tf.reset_default_graph()

# Feed-forward
weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights, 0)

# Feed the reward and chosen action
# into the network and compute the 
# loss and use it to update the network

reward_holder = tf.placeholder(shape = [1], dtype = tf.float32)
action_holder = tf.placeholder(shape = [1], dtype = tf.int32)
responsible_weight = tf.slice(weights, action_holder, [1])

loss = -(tf.log(responsible_weight) * reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
update = optimizer.minimize(loss)

# Train the agent
num_episodes = 1000
total_rewards = np.zeros(num_bandits)
e = 0.1

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
	sess.run(init)
	i = 0

	while i < num_episodes:
		# Choose an action
		if np.random.rand(1) < e:
			action = np.random.randint(num_bandits)
		else:
			action = sess.run(chosen_action)

		reward = pullBandit(bandits[action])

		# Update the network
		_, resp, ww = sess.run([update, responsible_weight, weights], 
			feed_dict = {reward_holder:[reward], action_holder:[action]})

		# Update running tally of scores
		total_rewards[action] += reward

		if i % 50 == 0:
			print("Running rewards for the " + str(num_bandits) + " bandits: " + str(total_rewards))
			i += 1

print("The agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promising....")
if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print("...and it was right!")
else:
    print("...and it was wrong!")
