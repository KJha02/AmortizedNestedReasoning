import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point, Line
import time
import random
from enum import Enum
import _pickle as pickle
import pdb
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

class Action(Enum):
	FORWARD = (0, 4000)
	LEFT = (0.5, 0)
	RIGHT = (-1.15, -1100)
	STOP = (0, -1e7)
	SIGNAL = (0, 0)

# class Action(Enum):
# 	FORWARD = (0, 4000)
# 	LEFT = (0.75, -300)
# 	RIGHT = (-0.69, -150)
# 	STOP = (0, -1e7)
# 	SIGNAL = (0, 0)


# class Action(Enum):
# 	FORWARD = (0, 5000)
# 	LEFT = (0.42, -50)
# 	RIGHT = (-0.9, -170)
# 	STOP = (0, -1e7)
# 	SIGNAL = (0, 0)


class Goal(Enum):
	# high level goals for each agent
	FORWARD = "forward"
	LEFT = "left"
	RIGHT = "right"


def isVisible(x, y, world, sight_threshold=np.inf):
	'''
	Returns whether the two cars are visible to each other
	'''
	
	try:  # if y is an agent
		line_of_sight = Line(x.center, y.center)
	except:
		line_of_sight = Line(x.center, y)
	if line_of_sight.length > sight_threshold:
		return False
		
	# Find the direction vectors
	direction_vector_heading = np.array([np.cos(x.heading), np.sin(x.heading)])
	direction_vector_line_of_sight = [line_of_sight.p2.x - line_of_sight.p1.x, line_of_sight.p2.y - line_of_sight.p1.y]

	# Calculate the dot product
	dot_product = np.dot(direction_vector_heading, direction_vector_line_of_sight)

	# Calculate the magnitude of each vector
	magnitude_heading = np.linalg.norm(direction_vector_heading)
	magnitude_line_of_sight = np.linalg.norm(direction_vector_line_of_sight)

	# Calculate the angle
	cos_theta = dot_product / (magnitude_heading * magnitude_line_of_sight)
	theta = np.arccos(cos_theta)

	observable_angle = 135
	#observable_angle = 45  # comment out if you don't want nearsighted driver
	if theta > np.pi * observable_angle/2/360:
		# The target falls outside the driver's field of view
		return False
			
	for obstructions in world.agents:
		# Ignore the sidewalks
		if isinstance(obstructions, Painting):
			continue

		if obstructions != x and obstructions != y:
			if line_of_sight.intersectsWith(obstructions.obj):
				# Obstructions in the line of sight
				return False
							
	return True

def get_partial_states(world, id=None, width=120, height=120):
	'''
	Returns partial observability of all cars in world
	specify agentID if you want a specific car's partial observation
	'''
	partial_states = []  # length should be number of cars in world
	for agent in world.dynamic_agents:
		if id is not None:
			if agent.ID != id:
				continue
		if type(agent) == Car:  # we only consider cars
			local_w = world.clone()
			local_w.dynamic_agents = []
			local_w.add(agent)  # we know where we are
			for other in world.dynamic_agents:
				if other != agent:
					if isVisible(agent, other, world):
						local_w.add(pickle.loads(pickle.dumps(other)))
			
			if id is not None:
				return local_w
			else:
				partial_states.append((agent.ID, local_w))
	return partial_states

class Scenario1:
	def __init__(self, human_controller=False, render=False, num_cars=3):
		self.human_controller = human_controller
		self.dt = 0.01           # time steps in terms of seconds
		self.controller = None
		self.world_size = 60
		self.goal_space = list(Goal)
		self.w = World(self.dt, width=self.world_size, height=self.world_size, ppm=6)
		self.init_world()
		self.initial_w = pickle.loads(pickle.dumps(self.w))
		self.render = render
		if self.render:
			self.w.render()
		self.step_number = 0

		if self.human_controller:
			if not self.controller:
				from interactive_controllers import SteeringWheelController
				self.controller = SteeringWheelController(self.w)
		self.num_cars = num_cars
		

	def init_world(self):
		# Add some sidewalks and RectangleBuildings
		lane_width = 6
		sidewalk_width = 2
		painting_size = self.world_size - lane_width * 2
		building_size = self.world_size - lane_width * 2 - sidewalk_width * 2
		self.w.add(Painting(Point(0, 0), Point(painting_size, painting_size), 'gray80'))
		self.w.add(Painting(Point(0, self.world_size), Point(painting_size, painting_size), 'gray80'))
		self.w.add(Painting(Point(self.world_size, 0), Point(painting_size, painting_size), 'gray80'))
		self.w.add(Painting(Point(self.world_size, self.world_size), Point(painting_size, painting_size), 'gray80'))

		self.w.add(RectangleBuilding(Point(0, 0), Point(building_size, building_size)))
		self.w.add(RectangleBuilding(Point(0, self.world_size), Point(building_size, building_size)))
		self.w.add(RectangleBuilding(Point(self.world_size, 0), Point(building_size, building_size)))
		self.w.add(RectangleBuilding(Point(self.world_size, self.world_size), Point(building_size, building_size)))

		# Add cars
		cars = [Car(Point(27,42), -np.pi/2),
				Car(Point(27,52), -np.pi/2),
				Car(Point(33,42), np.pi/2),
				Car(Point(33,52), np.pi/2),
				Car(Point(42,33), np.pi, 'blue'),
				Car(Point(52,33), np.pi, 'blue'),
				Car(Point(42,27), 0, 'blue'),
				Car(Point(52,27), 0, 'blue'),
				Car(Point(27,18), -np.pi/2),
				Car(Point(27,8), -np.pi/2),
				Car(Point(33,18), np.pi/2),
				Car(Point(33,8), np.pi/2),
				Car(Point(18,33), np.pi, 'blue'),
				Car(Point(8,33), np.pi, 'blue'),
				Car(Point(18,27), 0, 'blue'),
				Car(Point(8,27), 0, 'blue'),]

		self.initialLocMapping = {}
		for i, car in enumerate(cars):  # give ID
			car.ID = i
			self.initialLocMapping[i] = car.center

		car1_possible = [0, 1, 10, 11]
		car2_possible = [4, 5, 14, 15]
		all_possible = car1_possible + car2_possible

		# Sample two cars
		self.car_ids = [random.choice(car1_possible), random.choice(car2_possible)]
		# sub sample third agent
		all_possible.remove(self.car_ids[0])
		all_possible.remove(self.car_ids[1])
		for _ in range(self.num_cars - 2):
			new_car_id = random.choice(all_possible)
			self.car_ids.append(new_car_id)
			all_possible.remove(self.car_ids[-1])

		for c_id in self.car_ids:
			car = cars[c_id]
			self.sample_goal(car)
			# Sample initial velocity
			init_speed = 50 
			car.velocity = Point(init_speed, 0)
			self.w.add(car)


	def sample_goal(self, agent):
		agent.actionGoal = random.choice(self.goal_space)

	def reset(self):
		self.w.reset()
		# self.init_world()
		self.w = pickle.loads(pickle.dumps(self.initial_w))
		self.step_number = 0

	def default_step(self):
		self.w.tick()
		self.w.render()
		time.sleep(self.dt/400)       
		self.step_number += 1

	def get_done(self, state, timestep):
		if state.collision_exists():
			print("Detected crash")
			return True
		elif timestep > 100:
			return True
		else:
			return False

	def get_reward(self, state):
		reward = -1.0
		if state.collision_exists():
			reward += -100
		# TODO: Add some way to check if agents completed their goals
		return reward




	def step(self, controls: dict):
		"""
		Update the controls for c1 and c2 and steps forward in the simulation.
		
		Args:
			controls: a dictionary with car id's as keys and their control inputs as values.
				Example: {'c1': [c1_input_steering, c1_input_acceleration], ...}
		
		Returns:
			states: a dictionary where each key is a car ID and the corresponding value is
				a list of that car's updated state and its observed partial state.
				Example: {'c1': [self.c1.center, self.c1.velocity, c1_partial_state], ...}
		"""

		# Update control

		for car in self.w.dynamic_agents:
			if car.ID in controls:
				control = controls[car.ID]
				input_steering, input_acceleration = control.value
				car.set_control(input_steering, input_acceleration)

		# for car_id, control in controls.items():
		# 	input_steering, input_acceleration = control.value
		# 	if car_id == 'c1':
		# 		# if input_acceleration == -100:  # make stopping just stay in place
		# 		# 	self.c1.velocity = Point(0,0)
		# 		# 	self.c1.angular_velocity = 0
		# 		# else:
		# 		self.c1.set_control(input_steering, input_acceleration)
		# 	elif car_id == 'c2':
		# 		self.c2.set_control(input_steering, input_acceleration)



		# Step forward
		self.w.tick()
		if self.render:
			self.w.render()
			time.sleep(self.dt/400)       
		self.step_number += 1

		# print(f"{self.c1.center}")

		# Return updated states of cars, and their observed partial states
		partial_states = get_partial_states(self.w)
		next_states = {}
		for i, car in enumerate(self.w.dynamic_agents):
			next_states[car.ID] = [car.center, car.velocity, partial_states[i]]

		# next_states = {
		# 	'c1': [self.c1.center, self.c1.velocity, partial_states[0]],
		# 	'c2': [self.c2.center, self.c2.velocity, partial_states[1]]
		# }  
		reward = self.get_reward(self.w)
		done = self.get_done(self.w, self.step_number)
		return next_states, reward, done, {"state": self.w}

	def transition(self, state, controls: dict):
		"""
		Show the next state after taking an action but doesn’t impact the ground truth state of the world.
		
		Args:
			state: a World
			controls: a dictionary with car id's as keys and their control inputs as values.
				Example: {'c1': [c1_input_steering, c1_input_acceleration], ...}
		
		Returns:
			next_states: a dictionary where each key is a car ID and the corresponding value is
				a list of that car's expected next state.
				Example: {'c1': [c1_new_center, c1_new_velocity], ...}
		"""
		# update control
		next_state = pickle.loads(pickle.dumps(state))
		# prev_loc = {}
		for car in next_state.dynamic_agents:
			if car.ID in controls:
				control = controls[car.ID]
				input_steering, input_acceleration = control.value
				# if input_acceleration == -100:
				# 	car.velocity = Point(0,0)
				# 	car.angular_velocity = 0
				# else:
				car.set_control(input_steering, input_acceleration)
			# prev_loc[car.ID] = (car.center.x, car.center.y)

		
		next_state.tick()
		# new_locs = {}
		# for car in next_state.dynamic_agents:
		# 	new_locs[car.ID] = (car.center.x, car.center.y)
		# 	if car.ID in [4, 12, 0]:
		# 		assert (car.center.x, car.center.y) != prev_loc
		# pdb.set_trace()
		return next_state






		# next_states = {}
		# for car_id, agent in zip(['c1', 'c2'], [self.c1, self.c2]):
		# 	# Update control
		# 	if car_id in controls.keys():
		# 		new_inputSteering = controls[car_id].value[0]
		# 		new_inputAcceleration = controls[car_id].value[1]
		# 	else:
		# 		new_inputSteering = agent.inputSteering
		# 		new_inputAcceleration = agent.inputAcceleration
				
		# 	# Kinematic model dynamics
		# 	speed = agent.speed
		# 	heading = agent.heading
		# 	lr = agent.rear_dist
		# 	lf = lr
		# 	beta = np.arctan(lr / (lf + lr) * np.tan(new_inputSteering))
		# 	new_angular_velocity = speed * new_inputSteering
		# 	new_acceleration = new_inputAcceleration - agent.friction
		# 	new_speed = np.clip(speed + new_acceleration * self.dt, agent.min_speed, agent.max_speed)
		# 	new_heading = heading + ((speed + new_speed)/lr) * np.sin(beta) * self.dt / 2.
		# 	angle = (heading + new_heading)/2. + beta
		# 	new_center = agent.center + (speed + new_speed) * Point(np.cos(angle), np.sin(angle))*self.dt / 2.
		# 	new_velocity = Point(new_speed * np.cos(new_heading), new_speed * np.sin(new_heading))

		# 	next_states[car_id] = [new_center, new_velocity]

		# return next_states
