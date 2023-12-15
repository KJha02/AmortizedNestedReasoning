from entities import RectangleEntity, CircleEntity, RingEntity
from geometry import Point
from enum import Enum
import numpy as np

# For colors, we use tkinter colors. See http://www.science.smith.edu/dftwiki/index.php/Color_Charts_for_TKinter

class Action(Enum):
	FORWARD = (0, 4000)
	LEFT = (0.5, 0)
	RIGHT = (-1.15, -1100)
	STOP = (0, -1e7)
	SIGNAL = (0, 0)
# class Action(Enum):
# 	FORWARD = (0, 5000)
# 	LEFT = (0.42, -50)
# 	RIGHT = (-0.9, -170)
# 	STOP = (0, -1e7)
# 	SIGNAL = (0, 0)




# flat vector [a1_exists (one hot for agent ID), x, y, action (one hot for ) | a2_exists, x, y, action]
# [0,0,0,0] if a1 exists --> [1,0,0,0], a1.x, a1.y, one-hot of a1 action


# [forward prob, left prob, right prob] * 16



class Car(RectangleEntity):
	def __init__(self, center: Point, heading: float, color: str = 'red', ID=0, actionGoal=None):
		size = Point(4., 2.)
		movable = True
		friction = 0.06
		super(Car, self).__init__(center, heading, size, movable, friction)
		self.color = color
		self.collidable = True
		self.ID = ID
		self.prev_action = Action.FORWARD
		self.actionGoal = actionGoal
		
class Pedestrian(CircleEntity):
	def __init__(self, center: Point, heading: float, color: str = 'LightSalmon3'): # after careful consideration, I decided my color is the same as a salmon, so here we go.
		radius = 0.5
		movable = True
		friction = 0.2
		super(Pedestrian, self).__init__(center, heading, radius, movable, friction)
		self.color = color
		self.collidable = True
		
class RectangleBuilding(RectangleEntity):
	def __init__(self, center: Point, size: Point, color: str = 'gray26'):
		heading = 0.
		movable = False
		friction = 0.
		super(RectangleBuilding, self).__init__(center, heading, size, movable, friction)
		self.color = color
		self.collidable = True
		
class CircleBuilding(CircleEntity):
	def __init__(self, center: Point, radius: float, color: str = 'gray26'):
		heading = 0.
		movable = False
		friction = 0.
		super(CircleBuilding, self).__init__(center, heading, radius, movable, friction)
		self.color = color
		self.collidable = True

class RingBuilding(RingEntity):
	def __init__(self, center: Point, inner_radius: float, outer_radius: float, color: str = 'gray26'):
		heading = 0.
		movable = False
		friction = 0.
		super(RingBuilding, self).__init__(center, heading, inner_radius, outer_radius, movable, friction)
		self.color = color
		self.collidable = True

class Painting(RectangleEntity):
	def __init__(self, center: Point, size: Point, color: str = 'gray26', heading: float = 0.):
		movable = False
		friction = 0.
		super(Painting, self).__init__(center, heading, size, movable, friction)
		self.color = color
		self.collidable = False
