import math
import numpy as np
from honeycomb_rover_sim.utils.geometry import normalize_angle

class Robot:
    def __init__(self, config):
        self.cfg = config['robot']
        self.x, self.y, self.yaw = self.cfg['start_pose']
        self.v = 0.0
        self.w = 0.0
        self.width = self.cfg['width']
        self.length = self.cfg['length']
        self.radius = self.cfg['radius']
        
        self.max_v = self.cfg['max_v']
        self.max_w = self.cfg['max_w']

    def update(self, v, w, dt):
        # Limit inputs
        self.v = np.clip(v, -self.max_v, self.max_v)
        self.w = np.clip(w, -self.max_w, self.max_w)
        
        # Kinematics
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.w * dt
        self.yaw = normalize_angle(self.yaw)

    def get_pose(self):
        return self.x, self.y, self.yaw
