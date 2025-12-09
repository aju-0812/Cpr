import math
import numpy as np
from honeycomb_rover_sim.utils.geometry import normalize_angle

class Controller:
    def __init__(self, config):
        self.cfg = config['control']
        self.lookahead = self.cfg['lookahead']
        self.kp_v = self.cfg['kp_v']
        self.kp_w = self.cfg['kp_w']
        self.obstacle_check_dist = self.cfg['obstacle_check_dist']

    def compute_command(self, robot, path, arena):
        if not path:
            return 0.0, 0.0

        # 1. Find target point on path
        target_idx = self._find_target_index(robot, path)
        target = path[target_idx]
        
        # 2. Pure Pursuit
        dx = target['x'] - robot.x
        dy = target['y'] - robot.y
        alpha = normalize_angle(math.atan2(dy, dx) - robot.yaw)
        
        # Desired curvature
        # L = lookahead distance (dist to target)
        L = math.hypot(dx, dy)
        if L < 0.1: L = 0.1
        
        # gamma = 2*sin(alpha)/L
        # w = v * gamma
        
        target_v = target['v']
        
        # Simple P-controller for heading
        w = alpha * self.kp_w
        
        # 3. Local Avoidance (Simple DWA-like check)
        # Check for obstacles ahead
        if self._check_collision_ahead(robot, arena):
            # Emergency slow down and turn away
            target_v *= 0.1
            w += 1.0 # Bias turn left? Or away from nearest obstacle?
            # Ideally should check which side is clear. 
            # For this simple sim, we rely on global planner mostly.
            # Let's just slow down.
        
        return target_v, w

    def _find_target_index(self, robot, path):
        # Find point closest to lookahead distance
        min_dist = float('inf')
        idx = 0
        
        # Find closest point first
        closest_idx = 0
        closest_dist = float('inf')
        for i, p in enumerate(path):
            d = math.hypot(p['x'] - robot.x, p['y'] - robot.y)
            if d < closest_dist:
                closest_dist = d
                closest_idx = i
                
        # Look forward from there
        for i in range(closest_idx, len(path)):
            d = math.hypot(path[i]['x'] - robot.x, path[i]['y'] - robot.y)
            if d > self.lookahead:
                return i
        
        return len(path) - 1

    def _check_collision_ahead(self, robot, arena):
        # Cast a circle ahead
        check_x = robot.x + self.obstacle_check_dist * math.cos(robot.yaw)
        check_y = robot.y + self.obstacle_check_dist * math.sin(robot.yaw)
        
        # Check against obstacles
        if arena.is_in_collision(check_x, check_y, robot.radius):
            return True
        return False
