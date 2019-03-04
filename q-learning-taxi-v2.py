import gym
import numpy as np 
import random

# Create and show gym environment
env = gym.make("Taxi-v2")
env.render()

# Create Q-table and initialize
action_size = env.action_space.n 
print("Action Size: " + str(action_size))

state_size = env.observation_space.n 
print("State Size: " + str(state_size))

Q = np.zeros([state_size, action_size])

# Specify hyperparameters
num_episodes = 50000
test_episodes = 100
steps = 100

beta = 0.8
gamma = 0.98

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

# Q-learning algorithm
for episode in range(num_episodes):
	s = env.reset()
	step = 0
	done = False

	for step in range(steps):
		# Choose an action
		tradeoff = random.uniform(0, 1)

		if tradeoff > epsilon:
			a = np.argmax(Q[s, :])
		else:
			a = env.action_space.sample()

		# Take one step in the algorithm
		s_prime, r, done, _ = env.step(a)

		# Update Q-table
		Q[s, a] = Q[s,a] + beta * (r + gamma * np.max(Q[s_prime, :]) - Q[s,a])

		s = s_prime

		if done == True:
			break

	episode += 1

	# Reduce exploration with time
	epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

print(Q)

# Test the resulting Q-table
env.reset()
rewards = []

for episode in range(test_episodes):
	s = env.reset()
	step = 0
	done = False
	total_rewards = 0
	print("******************************************")
	print("EPISODE: ", episode)

	for step in range(steps):
		env.render()

		# Take action
		a = np.argmax(Q[s, :])

		# Step through the algorithm
		s_prime, r, done, _ = env.step(a)

		total_rewards += r

		if done:
			rewards.append(total_rewards)
			print("Score: ", total_rewards)
			break

		s = s_prime

env.close()
print("Score over time: " + str(sum(rewards) / test_episodes))