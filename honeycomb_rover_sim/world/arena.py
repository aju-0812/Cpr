import yaml
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Obstacle:
    x: float
    y: float
    r: float

@dataclass
class Zone:
    type: str
    # Union of rect, polygon, circle
    rect: List[float] = None
    polygon: List[List[float]] = None
    circle: Dict = None

class Arena:
    def __init__(self, config_source):
        if isinstance(config_source, str):
            with open(config_source, 'r') as f:
                self.cfg = yaml.safe_load(f)
        elif isinstance(config_source, dict):
            self.cfg = config_source
        else:
            raise ValueError("config_source must be a file path or a dictionary")
        
        self.width = self.cfg['arena']['width']
        self.height = self.cfg['arena']['height']
        
        self.obstacles = [Obstacle(**o) for o in self.cfg['obstacles']]
        self.zones = {}
        for name, data in self.cfg['zones'].items():
            self.zones[name] = Zone(**data)

    def is_in_collision(self, x, y, radius):
        # Check bounds
        if x < 0 or x > self.width or y < 0 or y > self.height:
            return True
            
        # Check obstacles
        for obs in self.obstacles:
            dist = np.hypot(x - obs.x, y - obs.y)
            if dist < (obs.r + radius):
                return True
        return False

    def get_clearance(self, x, y):
        """Distance to nearest obstacle or wall."""
        min_dist = min(x, self.width - x, y, self.height - y)
        
        for obs in self.obstacles:
            dist = np.hypot(x - obs.x, y - obs.y) - obs.r
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def get_zone_at(self, x, y):
        for name, zone in self.zones.items():
            if zone.type == 'start' and zone.rect:
                if zone.rect[0] <= x <= zone.rect[2] and zone.rect[1] <= y <= zone.rect[3]:
                    return name
            elif zone.type == 'danger' and zone.polygon:
                # Simple point in polygon check (ray casting)
                poly = zone.polygon
                n = len(poly)
                inside = False
                p1x, p1y = poly[0]
                for i in range(n + 1):
                    p2x, p2y = poly[i % n]
                    if y > min(p1y, p2y):
                        if y <= max(p1y, p2y):
                            if x <= max(p1x, p2x):
                                if p1y != p2y:
                                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                                if p1x == p2x or x <= xinters:
                                    inside = not inside
                    p1x, p1y = p2x, p2y
                if inside:
                    return name
            elif zone.type == 'safe' and zone.circle:
                dist = np.hypot(x - zone.circle['x'], y - zone.circle['y'])
                if dist <= zone.circle['r']:
                    return name
        return None
