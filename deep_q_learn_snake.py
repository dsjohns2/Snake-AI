from __future__ import print_function
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tkinter
import matplotlib.pyplot as plt

# Neural Network
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.num_input_params = 5
		self.num_classes = 1
		self.fc1 = nn.Linear(self.num_input_params, 120)
		self.fc2 = nn.Linear(120, 120)
		self.fc3 = nn.Linear(120, 60)
		self.fc4 = nn.Linear(60, self.num_classes)

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		x = F.relu(x)
		x = self.fc4(x)
		return x

# Train the Snake network
net = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=.0001)
dimensions = (5, 5)
for episode in range(0, 1000000):
	print("Episode: " + str(episode))
	snake_location = [(random.randint(0, dimensions[0]-1), random.randint(0, dimensions[1]-1))]
	free_space = []
	for i in range(dimensions[0]):
		for j in range(dimensions[1]):
			free_space.append((i, j))	
	free_space.remove(snake_location[0])
		
	alive = True
	eaten = True
	score = 0

	while(alive):
		# Spawn ball if needed
		if(eaten):
			ball_location = free_space[random.randint(0, len(free_space)-1)]
			free_space.remove(ball_location)
			eaten = False
		
		# Move Snake
		initial_state = snake_location[0]
		direction = random.randint(0, 3)
		if(direction == 0):
			snake_location = [(snake_location[0][0]-1, snake_location[0][1])] + snake_location
		elif(direction == 1):
			snake_location = [(snake_location[0][0], snake_location[0][1]+1)] + snake_location
		elif(direction == 2):
			snake_location = [(snake_location[0][0]+1, snake_location[0][1])] + snake_location
		else:
			snake_location = [(snake_location[0][0], snake_location[0][1]-1)] + snake_location
		
		free_space.append(snake_location[-1])
		del(snake_location[-1])
		new_state = snake_location[0]
		
		# Check if died or eaten
		if(snake_location[0] in free_space):
			free_space.remove(snake_location[0])
			reward = -1
		elif(snake_location[0] == ball_location):
			eaten = True
			reward = 10
			score += 100
		else:
			reward = -100
			alive = False

		action = direction
		lr = .5
		dv = .5
		if(reward == -100):
			optimizer.zero_grad()
			X = np.array([initial_state[0], initial_state[1], ball_location[0], ball_location[1], action])
			X = X.astype(np.float32)
			X = torch.from_numpy(X)
			q_guess = net(X)
			q_target = (1 - lr) * net(X) + lr * reward
			loss = criterion(q_guess, q_target.detach())
			loss.backward()
			optimizer.step()
		else:
			optimizer.zero_grad()
			X = np.array([initial_state[0], initial_state[1], ball_location[0], ball_location[1], action])
			X = X.astype(np.float32)
			X = torch.from_numpy(X)
			new_X_actions = np.array([[new_state[0], new_state[1], ball_location[0], ball_location[1], 0], [new_state[0], new_state[1], ball_location[0], ball_location[1], 1], [new_state[0], new_state[1], ball_location[0], ball_location[1], 2], [new_state[0], new_state[1], ball_location[0], ball_location[1], 3]])
			new_X_actions = new_X_actions.astype(np.float32)
			new_X_actions = torch.from_numpy(new_X_actions)
			new_q_vals = net(new_X_actions)
			new_q_vals = new_q_vals.detach().numpy()
			q_guess = net(X)
			q_target = (1 - lr) * net(X) + lr * (reward + dv * np.amax(new_q_vals))
			loss = criterion(q_guess, q_target.detach())
			loss.backward()
			optimizer.step()


# Test the Snake
snake_location = [(random.randint(0, dimensions[0]-1), random.randint(0, dimensions[1]-1))]
free_space = []
for i in range(dimensions[0]):
	for j in range(dimensions[1]):
		free_space.append((i, j))	
free_space.remove(snake_location[0])
	
alive = True
eaten = True
score = 0
#####
for i in range(5):
	for j in range(5):
		print("(", end=" ")
		for k in range(4):
			X = np.array([i, j, 0, 0, k])
			X = X.astype(np.float32)
			X = torch.from_numpy(X)
			print(round(float(net(X)), 3), end=" ")
		print(")", end=" ")
	print()
#####
while(alive):
	# Spawn ball if needed
	if(eaten):
		ball_location = free_space[random.randint(0, len(free_space)-1)]
		free_space.remove(ball_location)
		eaten = False

	# Print game
	for i in range(dimensions[1]+2):
		print("-", end='')
	print()
	for i in range(dimensions[0]):
		print("|", end='')
		for j in range(dimensions[1]):
			space = (i, j)
			if(ball_location == space):
				print("0", end='')
			elif(space in free_space):
				print(" ", end='')
			else:
				print("X", end='')
		print("|", end='')
		print()
	for i in range(dimensions[1]+2):
		print("-", end='')
	print()
	print(score)
	
	# Move Snake
	X_actions = np.array([[snake_location[0][0], snake_location[0][1], ball_location[0], ball_location[1], 0], [snake_location[0][0], snake_location[0][1], ball_location[0], ball_location[1], 1], [snake_location[0][0], snake_location[0][1], ball_location[0], ball_location[1], 2], [snake_location[0][0], snake_location[0][1], ball_location[0], ball_location[1], 3]])
	X_actions = X_actions.astype(np.float32)
	X_actions = torch.from_numpy(X_actions)
	q_vals = net(X_actions)
	q_vals = q_vals.detach().numpy()
	direction = np.argmax(q_vals)
	print(q_vals)
	if(direction == 0):
		snake_location = [(snake_location[0][0]-1, snake_location[0][1])] + snake_location
	elif(direction == 1):
		snake_location = [(snake_location[0][0], snake_location[0][1]+1)] + snake_location
	elif(direction == 2):
		snake_location = [(snake_location[0][0]+1, snake_location[0][1])] + snake_location
	else:
		snake_location = [(snake_location[0][0], snake_location[0][1]-1)] + snake_location
	
	free_space.append(snake_location[-1])
	del(snake_location[-1])
	
	# Check if died or eaten
	if(snake_location[0] in free_space):
		free_space.remove(snake_location[0])
	elif(snake_location[0] == ball_location):
		eaten = True
		score += 100
	else:
		print("Dead")
		print()
		print("Final Score: " + str(score))
		alive = False
	time.sleep(.5)
