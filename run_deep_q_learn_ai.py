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

# Load the Neural Net
net = torch.load("deep_q_net.pt")

# Run Snake
dimensions = (5, 5)
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
