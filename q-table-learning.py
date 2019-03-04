# Import the GYM environment
import gym
import numpy as np 

# Load environment
env = gym.make('FrozenLake-v0')

# Print states and actions info
print("Num States: " + str(env.observation_space.n))
print("Num Actions: " + str(env.action_space.n))

# Q-Table learning algorithm
Q = np.zeros([env.observation_space.n, env.action_space.n])
beta = 0.8
gamma = 0.98
num_episodes = 2000

rewardList = []

for i in range(num_episodes):
	# Reset environment and get new observation
	s = env.reset()
	rSum = 0
	d = False
	j = 0

	# Q-Table learning
	while j < 1000:
		j += 1

		# Choose an action from the Q-table
		a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n) 
			* (1./(i+1)))

		# Get new state and reward
		s_prime, r, d, _ = env.step(a)

		# Update Q-table
		Q[s,a] = Q[s,a] + beta * (r + gamma * np.max(Q[s_prime, :]) - Q[s,a])
		rSum += r
		s = s_prime

		if d == True:
			break

	rewardList.append(rSum)

print("Score over time" + str(sum(rewardList) / num_episodes))

print("Final Q-Table values: ")
print(Q)
