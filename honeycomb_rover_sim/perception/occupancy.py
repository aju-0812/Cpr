import numpy as np

class OccupancyGrid:
    def __init__(self, width, height, resolution=0.1):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.cols = int(width / resolution)
        self.rows = int(height / resolution)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int8) # 0: unknown, 1: free, 100: occupied

    def update_from_sim(self, arena):
        """Cheat method to populate grid from ground truth for visualization."""
        # In a real sim, this would use raycasting
        pass
