from agents import Car, Pedestrian, RectangleBuilding
from geometry import Line, Point
from entities import Entity
from typing import Union
import _pickle as pickle
from pathlib import Path
from PIL import Image as NewImage
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

class World:
    def __init__(self, dt: float, width: float, height: float, ppm: float = 8):
        self.dynamic_agents = []
        self.static_agents = []
        self.t = 0 # simulation time
        self.dt = dt # simulation time step
        self.width = width
        self.height = height
        self.ppm= ppm
        self.visualizer = None
        
    def add(self, entity: Entity):
        if entity.movable:
            self.dynamic_agents.append(entity)
        else:
            self.static_agents.append(entity)
        
    def tick(self):
        for agent in self.dynamic_agents:
            agent.tick(self.dt)
        self.t += self.dt

    def clone(self):
        new_world = World(self.dt, self.width, self.height, self.ppm)
        new_world.dynamic_agents = pickle.loads(pickle.dumps(self.dynamic_agents))
        new_world.static_agents = pickle.loads(pickle.dumps(self.static_agents))
        new_world.t = self.t 
        return new_world
    
    def render(self, save_dir=None, timestep=0):
        self.visualizer.create_window(bg_color = 'gray')
        self.visualizer.update_agents(self.agents)
        if save_dir is not None:
            path_eps = f"{save_dir}/{timestep}_gui.eps"
            path_gui = f"{save_dir}/{timestep}_gui.png"
            Path(path_gui).parent.mkdir(parents=True, exist_ok=True)
            self.visualizer.win.postscript(file=path_eps)
            img = NewImage.open(path_eps)
            img.save(path_gui, "png")
    def agent_speed(self, agentID):
        for a in self.dynamic_agents:
            if a.ID == agentID:
                return a.speed

    def agent_exists(self, agentID):
        for a in self.dynamic_agents:
            if a.ID == agentID:
                return a
        return None

    def agent_near_wall(self, agentID):
        ax, ay = None, None
        for a in self.dynamic_agents:
            if a.ID == agentID:
                ax, ay = a.center.x, a.center.y
                break
        horizontalLine = Line(Point(0, ay), Point(self.width, ay))
        verticalLine = Line(Point(ax, 0), Point(ax, self.height))
        for b in self.static_agents:
            if b.collidable:  # only consider walls not sidewalks
                if horizontalLine.intersectsWith(b.obj) or verticalLine.intersectsWith(b.obj):
                    return True
        return False
        
    @property
    def agents(self):
        return self.static_agents + self.dynamic_agents
        
    def collision_exists(self, agent = None):
        if agent is None:
            for i in range(len(self.dynamic_agents)):
                for j in range(i+1, len(self.dynamic_agents)):
                    if self.dynamic_agents[i].collidable and self.dynamic_agents[j].collidable:
                        if self.dynamic_agents[i].collidesWith(self.dynamic_agents[j]):
                            return True
                for j in range(len(self.static_agents)):
                    if self.dynamic_agents[i].collidable and self.static_agents[j].collidable:
                        if self.dynamic_agents[i].collidesWith(self.static_agents[j]):
                            return True
            return False
            
        if not agent.collidable: return False
        
        for i in range(len(self.agents)):
            if self.agents[i] is not agent and self.agents[i].collidable and agent.collidesWith(self.agents[i]):
                return True
        return False
    
    def close(self):
        self.reset()
        self.static_agents = []
        if self.visualizer.window_created:
            self.visualizer.close()
        
    def reset(self):
        self.dynamic_agents = []
        self.t = 0