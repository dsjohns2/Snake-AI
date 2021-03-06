from __future__ import print_function
import random
import time
import numpy as np

dimensions = (5, 5)
q_table = np.zeros((dimensions[0], dimensions[1], dimensions[0], dimensions[1], 4))
for episode in range(0, 100000):
	print(episode)
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
			print("Dead")
			print()
			print("Final Score: " + str(score))
			alive = False

		lr = .5
		dv = .5
		action = direction
		if(reward == -100):
			q_table[initial_state[0]][initial_state[1]][ball_location[0]][ball_location[1]][action] = (1 - lr) * q_table[initial_state[0]][initial_state[1]][ball_location[0]][ball_location[1]][action] + lr * (reward)
		else:
			q_table[initial_state[0]][initial_state[1]][ball_location[0]][ball_location[1]][action] = (1 - lr) * q_table[initial_state[0]][initial_state[1]][ball_location[0]][ball_location[1]][action] + lr * (reward + dv * np.amax(q_table[new_state[0]][new_state[1]][ball_location[0]][ball_location[1]]))


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
	direction = np.argmax(q_table[snake_location[0][0]][snake_location[0][1]][ball_location[0]][ball_location[1]])
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
