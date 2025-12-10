import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, RegularPolygon
from matplotlib.animation import FuncAnimation
import numpy as np
from honeycomb_rover_sim.utils.geometry import HexUtils

class Visualizer:
    def __init__(self, arena, grid, robot, config):
        self.arena = arena
        self.grid = grid
        self.robot = robot
        self.cfg = config['sim']['visualization']
        
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.ax.set_xlim(-0.5, arena.width + 0.5)
        self.ax.set_ylim(-0.5, arena.height + 0.5)
        self.ax.set_aspect('equal')
        
        self.robot_patch = None
        self.path_line = None
        self.refined_path_line = None
        self.scan_lines = []
        
        self._init_static_elements()

    def _init_static_elements(self):
        # Draw Hex Grid (Importance)
        # We can't draw thousands of polygons efficiently in animation loop
        # Draw them once as a collection or just draw the centers
        # Let's draw a heatmap using scatter
        nodes = list(self.grid.nodes.values())
        x = [n.x for n in nodes]
        y = [n.y for n in nodes]
        c = [n.importance for n in nodes]
        
        self.ax.scatter(x, y, c=c, cmap='coolwarm', s=20, alpha=0.3, vmin=-1, vmax=1)
        
        # Draw Obstacles
        for obs in self.arena.obstacles:
            circle = Circle((obs.x, obs.y), obs.r, color='black', alpha=0.7)
            self.ax.add_patch(circle)
            
        # Draw Zones
        for name, zone in self.arena.zones.items():
            if zone.type == 'start':
                rect = Rectangle((zone.rect[0], zone.rect[1]), 
                                 zone.rect[2]-zone.rect[0], 
                                 zone.rect[3]-zone.rect[1], 
                                 color='green', alpha=0.2, label='Start')
                self.ax.add_patch(rect)
            elif zone.type == 'danger':
                poly = Polygon(zone.polygon, color='red', alpha=0.2, label='Excavation')
                self.ax.add_patch(poly)
            elif zone.type == 'safe':
                circle = Circle((zone.circle['x'], zone.circle['y']), 
                                zone.circle['r'], color='blue', alpha=0.2, label='Construction')
                self.ax.add_patch(circle)

        # Robot
        self.robot_patch = Rectangle((0,0), self.robot.width, self.robot.length, 
                                     angle=0, color='orange', alpha=0.9)
        self.ax.add_patch(self.robot_patch)
        
        # Path lines
        self.path_line, = self.ax.plot([], [], 'g--', alpha=0.5, linewidth=2, label='A* Path')
        self.refined_path_line, = self.ax.plot([], [], 'c-', linewidth=2, label='Refined Path')
        
        # Pre-allocate scan lines (e.g., max 10 detections)
        self.scan_lines = []
        for _ in range(10):
            line, = self.ax.plot([], [], 'r-', alpha=0.5, linewidth=1)
            self.scan_lines.append(line)
        
        self.ax.legend(loc='upper right')

    def update(self, robot, path, refined_path, detected_markers):
        # Update Robot
        cx, cy, theta = robot.x, robot.y, robot.yaw
        w, h = robot.width, robot.length
        
        dx = -w/2 * np.cos(theta) - -h/2 * np.sin(theta)
        dy = -w/2 * np.sin(theta) + -h/2 * np.cos(theta)
        
        self.robot_patch.set_xy((cx + dx, cy + dy))
        self.robot_patch.angle = np.degrees(theta)
        
        # Update Paths
        if path:
            px = [n.x for n in path]
            py = [n.y for n in path]
            self.path_line.set_data(px, py)
            
        if refined_path:
            rpx = [p['x'] for p in refined_path]
            rpy = [p['y'] for p in refined_path]
            self.refined_path_line.set_data(rpx, rpy)

        # Update Scan Lines
        # Hide all first
        for line in self.scan_lines:
            line.set_data([], [])
            
        # Show active ones
        for i, det in enumerate(detected_markers):
            if i >= len(self.scan_lines): break
            # Line from robot to marker
            # We need marker position. 
            # Since we don't have marker map here, we can calculate from range/bearing
            # mx = rx + r * cos(yaw + bearing)
            mx = cx + det['range'] * np.cos(theta + det['bearing'])
            my = cy + det['range'] * np.sin(theta + det['bearing'])
            self.scan_lines[i].set_data([cx, mx], [cy, my])
        
        return [self.robot_patch, self.path_line, self.refined_path_line] + self.scan_lines

    def show(self):
        plt.show()
