import numpy as np
import matplotlib.pyplot as plt

class OccupancyGrid:
    def __init__(self, width=100, height=100, resolution=1.0):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.cols = int(width / resolution)
        self.rows = int(height / resolution)
        
        self.grid = np.zeros((self.rows, self.cols))
        
        self.lo_occ = np.log(0.8 / 0.2)
        self.lo_free = np.log(0.3 / 0.7)
        self.lo_max = 10.0
        self.lo_min = -10.0

    def world_to_grid(self, x, y):
        gx = int(x / self.resolution)
        gy = int(y / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = gx * self.resolution + self.resolution / 2
        y = gy * self.resolution + self.resolution / 2
        return x, y

    def is_valid(self, gx, gy):
        return 0 <= gx < self.cols and 0 <= gy < self.rows

    def update_map(self, rover_x, rover_y, scan_points):
        start_gx, start_gy = self.world_to_grid(rover_x, rover_y)
        
        if not self.is_valid(start_gx, start_gy):
            return

        for px, py in scan_points:
            end_gx, end_gy = self.world_to_grid(px, py)
            
            self.bresenham(start_gx, start_gy, end_gx, end_gy)
            
            if self.is_valid(end_gx, end_gy):
                self.grid[end_gy, end_gx] += self.lo_occ
                
        np.clip(self.grid, self.lo_min, self.lo_max, out=self.grid)

    def bresenham(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            if x0 == x1 and y0 == y1:
                break
            
            if self.is_valid(x0, y0):
                self.grid[y0, x0] += self.lo_free
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def get_probability_map(self):
        return 1.0 / (1.0 + np.exp(-self.grid))

    def plot(self):
        plt.imshow(self.get_probability_map(), cmap='Greys', origin='lower', 
                   extent=[0, self.width, 0, self.height], vmin=0, vmax=1)
        plt.colorbar(label='Occupancy Probability')
