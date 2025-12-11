import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height
        self.obstacles = []

    def add_obstacle(self, x, y, radius):
        self.obstacles.append({'type': 'circle', 'x': x, 'y': y, 'r': radius})

    def add_rectangle_obstacle(self, x, y, w, h):
        self.obstacles.append({'type': 'rect', 'x': x, 'y': y, 'w': w, 'h': h})
        
    def add_line_obstacle(self, x1, y1, x2, y2):
        self.obstacles.append({'type': 'line', 'p1': (x1, y1), 'p2': (x2, y2)})
        
    def add_polygon_obstacle(self, points):
        # points: list of (x, y) tuples
        self.obstacles.append({'type': 'polygon', 'points': points})

class Rover:
    def __init__(self, x=10, y=10, theta=0):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = 0.0
        self.w = 0.0
        self.dt = 0.1
        self.history = []

    def move(self, v, w):
        self.v = v
        self.w = w
        
        self.x += v * np.cos(self.theta) * self.dt
        self.y += v * np.sin(self.theta) * self.dt
        self.theta += w * self.dt
        
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        self.history.append((self.x, self.y))

class Sensor:
    def __init__(self, max_range=20, num_rays=360, fov=np.pi*2):
        self.max_range = max_range
        self.num_rays = num_rays
        self.fov = fov
        self.noise_std = 0.1

    def scan(self, env, rover_pose):
        rx, ry, rtheta = rover_pose
        angles = np.linspace(rtheta - self.fov/2, rtheta + self.fov/2, self.num_rays)
        ranges = []
        points = []

        for angle in angles:
            dist = self.raycast(env, rx, ry, angle)
            if dist < self.max_range:
                dist += np.random.normal(0, self.noise_std)
            
            ranges.append(dist)
            
            if dist < self.max_range:
                px = rx + dist * np.cos(angle)
                py = ry + dist * np.sin(angle)
                points.append([px, py])
        
        return np.array(ranges), np.array(points)

    def raycast(self, env, x, y, angle):
        min_dist = self.max_range
        
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # Ray end point (max range)
        rx2 = x + dx * self.max_range
        ry2 = y + dy * self.max_range

        for obs in env.obstacles:
            dist = float('inf')
            
            if obs['type'] == 'circle':
                dist = self.intersect_circle(x, y, dx, dy, obs)
            elif obs['type'] == 'line':
                dist = self.intersect_line(x, y, rx2, ry2, obs['p1'], obs['p2'])
            elif obs['type'] == 'polygon':
                dist = self.intersect_polygon(x, y, rx2, ry2, obs['points'])
            elif obs['type'] == 'rect':
                # Treat rect as polygon
                ox, oy, w, h = obs['x'], obs['y'], obs['w'], obs['h']
                points = [(ox, oy), (ox+w, oy), (ox+w, oy+h), (ox, oy+h)]
                dist = self.intersect_polygon(x, y, rx2, ry2, points)
                
            if dist < min_dist:
                min_dist = dist
                        
        return min_dist

    def intersect_circle(self, x, y, dx, dy, obs):
        ox, oy, r = obs['x'], obs['y'], obs['r']
        fx = x - ox
        fy = y - oy
        a = dx**2 + dy**2
        b = 2 * (fx*dx + fy*dy)
        c = (fx**2 + fy**2) - r**2
        discriminant = b**2 - 4*a*c
        
        if discriminant >= 0:
            t1 = (-b - np.sqrt(discriminant)) / (2*a)
            t2 = (-b + np.sqrt(discriminant)) / (2*a)
            if t1 > 0: return t1
            if t2 > 0: return t2
        return float('inf')

    def intersect_line(self, x1, y1, x2, y2, p3, p4):
        # Ray: (x1, y1) -> (x2, y2)
        # Segment: p3 -> p4
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0: return float('inf')
        
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
        
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            return ua * np.hypot(x2 - x1, y2 - y1) # Return distance
        return float('inf')

    def intersect_polygon(self, x1, y1, x2, y2, points):
        min_d = float('inf')
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            d = self.intersect_line(x1, y1, x2, y2, p1, p2)
            if d < min_d:
                min_d = d
        return min_d

if __name__ == "__main__":
    env = Environment()
    env.add_obstacle(30, 30, 5)
    env.add_obstacle(60, 40, 8)
    
    rover = Rover(x=10, y=10)
    sensor = Sensor()
    
    rover.move(1.0, 0.1)
    
    ranges, points = sensor.scan(env, (rover.x, rover.y, rover.theta))
    
    plt.figure(figsize=(10, 10))
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    
    for obs in env.obstacles:
        circle = plt.Circle((obs['x'], obs['y']), obs['r'], color='r')
        plt.gca().add_patch(circle)
        
    plt.plot(rover.x, rover.y, 'bo', markersize=10)
    plt.arrow(rover.x, rover.y, 2*np.cos(rover.theta), 2*np.sin(rover.theta), head_width=1)
    
    if len(points) > 0:
        plt.plot(points[:, 0], points[:, 1], 'g.', markersize=2)
        
    plt.title("Simulation Test")
    plt.show()
