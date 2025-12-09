import numpy as np

class RoverEnv:
    def __init__(self, grid_size=(20, 20)):
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size, dtype=int)
        self.start_pos = (0, 0)
        self.goal_pos = (grid_size[0]-1, grid_size[1]-1)
        self.rover_pos = self.start_pos
        
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = [0, 1, 2, 3]

    def reset(self):
        self.rover_pos = self.start_pos
        return self.rover_pos

    def set_obstacles(self, obstacles):
        """
        obstacles: list of (x, y) tuples
        """
        self.grid.fill(0)
        for x, y in obstacles:
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                self.grid[y, x] = 1 # 1 is obstacle

    def step(self, action):
        """
        Apply action and return (next_state, reward, done)
        """
        x, y = self.rover_pos
        
        if action == 0:   # Up
            y -= 1
        elif action == 1: # Down
            y += 1
        elif action == 2: # Left
            x -= 1
        elif action == 3: # Right
            x += 1
            
        # Check boundaries
        x = max(0, min(x, self.grid_size[0] - 1))
        y = max(0, min(y, self.grid_size[1] - 1))
        
        next_pos = (x, y)
        
        # Rewards
        if next_pos == self.goal_pos:
            reward = 100
            done = True
        elif self.grid[y, x] == 1: # Obstacle
            reward = -100
            done = True # Collision ends episode (or could just be a penalty)
            # Ensure we don't actually move into the wall
            next_pos = self.rover_pos 
        else:
            reward = -1 # Step penalty to encourage shortest path
            done = False
            
        self.rover_pos = next_pos
        return next_pos, reward, done
