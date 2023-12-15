import numpy as np
from world import World
import pickle
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point, Line
import time
import random
import geometry


def generate_single_driving_objects(width, height, b_width, b_height):
	'''
	Generates building locations and single agent with goal location for driving environment
	'''


	a_width = width // b_width  # artificial width and height for scaled down gridworld
	a_height = height // b_height

	assert a_width % 2 == 1 and a_width >= 3
	assert a_height % 2 == 1 and a_height >= 3

	# Use these characters for displaying the maze:
	EMPTY = ' '
	MARK = '@'
	WALL = chr(9608) # Character 9608 is '█'
	NORTH, SOUTH, EAST, WEST = 'n', 's', 'e', 'w'
	rotations = [np.pi / 2, -np.pi / 2, 0, np.pi]

	maze = {}  # Create the filled-in maze data structure to start:
	for x in range(a_width):
		for y in range(a_height):
			maze[(x, y)] = WALL  # Every space is a wall at first.

	def visit(x, y):
		""""Carve out" empty spaces in the maze at x, y and then
		recursively move to neighboring unvisited spaces. This
		function backtracks when the mark has reached a dead end."""
		maze[(x, y)] = EMPTY # "Carve out" the space at x, y.
		# printMaze(maze, x, y) # Display the maze as we generate it.
		# print('\n\n')

		while True:
			# Check which neighboring spaces adjacent to
			# the mark have not been visited already:
			unvisitedNeighbors = []
			if y > 1 and (x, y - 2) not in hasVisited:
				unvisitedNeighbors.append(NORTH)

			if y < a_height - 2 and (x, y + 2) not in hasVisited:
				unvisitedNeighbors.append(SOUTH)

			if x > 1 and (x - 2, y) not in hasVisited:
				unvisitedNeighbors.append(WEST)

			if x < a_width - 2 and (x + 2, y) not in hasVisited:
				unvisitedNeighbors.append(EAST)

			if len(unvisitedNeighbors) == 0:
				# BASE CASE
				# All neighboring spaces have been visited, so this is a
				# dead end. Backtrack to an earlier space:
				return
			else:
				# RECURSIVE CASE
				# Randomly pick an unvisited neighbor to visit:
				nextIntersection = random.choice(unvisitedNeighbors)

				# Move the mark to an unvisited neighboring space:

				if nextIntersection == NORTH:
					nextX = x
					nextY = y - 2
					maze[(x, y - 1)] = EMPTY # Connecting hallway.
				elif nextIntersection == SOUTH:
					nextX = x
					nextY = y + 2
					maze[(x, y + 1)] = EMPTY # Connecting hallway.
				elif nextIntersection == WEST:
					nextX = x - 2
					nextY = y
					maze[(x - 1, y)] = EMPTY # Connecting hallway.
				elif nextIntersection == EAST:
					nextX = x + 2
					nextY = y
					maze[(x + 1, y)] = EMPTY # Connecting hallway.

				hasVisited.append((nextX, nextY)) # Mark as visited.
				visit(nextX, nextY) # Recursively visit this space.


	# Carve out the paths in the maze data structure:
	hasVisited = [(1, 1)] # Start by visiting the top-left corner.
	visit(1, 1)

	# used to convert scaled down maze coordinates up to full world coordinates
	maze2buildX = lambda x: (x * b_width) + (b_width / 2)
	maze2buildY = lambda y: (y * b_height) + (b_height / 2)

	# iterate through maze and return buildings
	buildings = set()
	freeSpace = []
	for x in range(a_width):
		for y in range(a_height):
			if maze[(x, y)] == WALL:
				bx, by = maze2buildX(x), maze2buildY(y)
				buildings.add(RectangleBuilding(Point(bx, by), Point(b_width, b_height)))
			else:
				freeSpace.append((x, y))
	# pick two squares that are empty
	freeSampled = random.sample(freeSpace, 8)  # number of agents * 2

	heading_angles = []
	for sID, loc in enumerate(freeSampled):
		heading_angle = 0
		noWalls = []
		x, y = loc
		for i, shift in enumerate([(0, 1), (0, -1), (1, 0), (-1, 0)]):
			neighbor = (x + shift[0], y + shift[1])
			if neighbor in maze:  
				if maze[neighbor] != WALL:  # if there's no wall there, we can begin facing that direction
					noWalls.append(rotations[i])

		heading_angle = random.choice(noWalls)
		heading_angles.append(heading_angle)

		freeSampled[sID] = (maze2buildX(x), maze2buildY(y)) # convert to full world coords



	# agent_loc, target_loc = freeSampled[0], freeSampled[1]
	# agent_loc = 
	# target_loc = (maze2buildX(target_loc[0]), maze2buildY(target_loc[1]))
	# freeSampled = [agent_loc, target_loc]

	return buildings, freeSampled, heading_angles, maze

def generate_world(dt=0.1, width=135, height=135, b_width=9, b_height=9):
	w = World(dt, width, height, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.



	buildings, freeSampled, heading_angles, maze = generate_single_driving_objects(width, height, b_width, b_height)
	for b in buildings:
		w.add(b)

	for i, loc in enumerate(freeSampled):
		x, y = loc
		heading_angle = heading_angles[i]
		sx = b_width / 4
		sy = b_height / 4
		if heading_angle == np.pi / 2:  # if we face north, shift right to right lane
			x += sx
		elif heading_angle == -np.pi / 2:  # if we face south, shift left to right lane
			x -= sx
		elif heading_angle == np.pi:  # if we face west, shift up to right lane
			y += sy
		else:
			y -= sy
		freeSampled[i] = (x, y)

	colors = ['blue', 'green', 'red', 'orange']
	for i in range(0, len(freeSampled), 2):
		(ax, ay), (tx, ty) = freeSampled[i], freeSampled[i + 1]

		# this is the agent
		agent = Car(Point(ax, ay), heading_angles[i], color=colors[i//2])
		w.add(agent)

		# this is the goal
		w.add(Painting(Point(tx, ty), Point(1,1), color=colors[i//2]))

	return w

def isVisible(x, y, world):
	'''
	Returns whether the two cars are visible to each other
	'''
	try:  # if y is an agent
		line_of_sight = Line(x.center, y.center)
	except:
		line_of_sight = Line(x.center, y)
	for obstructions in world.agents:
		if obstructions != x and obstructions != y:
			if line_of_sight.intersectsWith(obstructions.obj):
				return False
	return True

def get_partial_states(world, width=120, height=120):
	'''
	Returns partial observability of all cars in world
	'''
	partial_states = []  # length should be number of cars in world
	for agent in world.dynamic_agents:
		if type(agent) == Car:  # we only consider cars
			local_w = World(world.dt, width, height, ppm = 6)  # copy time and static agents
			local_w.static_agents = pickle.loads(pickle.dumps(world.static_agents))
			local_w.t = world.t
			local_w.add(agent)  # we know where we are
			for other in world.dynamic_agents:
				if other != agent:
					if isVisible(agent, other, world):
						local_w.add(pickle.loads(pickle.dumps(other)))
			partial_states.append((agent.ID, local_w))
			
	return partial_states



if __name__ == "__main__":
	width = 99
	height = 99
	w = generate_world(width = width, height = height)


	w.render()
	for i in range(6):
		time.sleep(1)
	partial_states = get_partial_states(w, width, height)
	for i, p in enumerate(partial_states):
		print(f"Rendering for agent {i}")
		p.render()
		for i in range(10):
			time.sleep(1)
		p.close()
	w.close()
