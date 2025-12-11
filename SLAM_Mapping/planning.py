import numpy as np
import heapq

class GlobalPlanner:
    def __init__(self, grid):
        self.grid = grid

    def heuristic(self, a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def plan(self, start, goal):
        start_node = self.grid.world_to_grid(*start)
        goal_node = self.grid.world_to_grid(*goal)

        if not self.grid.is_valid(*start_node) or not self.grid.is_valid(*goal_node):
            print("Start or Goal is invalid")
            return []

        open_set = []
        heapq.heappush(open_set, (0, start_node))
        
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self.heuristic(start_node, goal_node)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_node:
                return self.reconstruct_path(came_from, current)

            neighbors = [
                (0, 1), (0, -1), (1, 0), (-1, 0),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self.grid.is_valid(*neighbor):
                    continue
                
                if self.grid.get_probability_map()[neighbor[1], neighbor[0]] > 0.5:
                    continue

                tentative_g_score = g_score[current] + np.sqrt(dx**2 + dy**2)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        print("No path found")
        return []

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        
        world_path = []
        for node in total_path[::-1]:
            world_path.append(self.grid.grid_to_world(*node))
        return world_path

class LocalPlanner:
    def __init__(self):
        self.max_speed = 1.0
        self.min_speed = 0.0
        self.max_yaw_rate = 40.0 * np.pi / 180.0
        self.max_accel = 0.2
        self.max_dyaw_rate = 40.0 * np.pi / 180.0
        self.v_reso = 0.1
        self.yaw_rate_reso = 0.1 * np.pi / 180.0
        self.dt = 0.1
        self.predict_time = 3.0
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_radius = 1.0

    def plan(self, x, y, theta, v, w, goal, obstacles):
        dw = self.calc_dynamic_window(v, w)
        
        best_u = [0.0, 0.0]
        min_cost = float('inf')
        all_trajectories = []
        
        for tv in np.arange(dw[0], dw[1], self.v_reso):
            for tw in np.arange(dw[2], dw[3], self.yaw_rate_reso):
                trajectory = self.predict_trajectory(x, y, theta, tv, tw)
                
                to_goal_cost = self.calc_to_goal_cost(trajectory, goal)
                speed_cost = self.speed_cost_gain * (self.max_speed - trajectory[-1, 3])
                ob_cost = self.calc_obstacle_cost(trajectory, obstacles)
                
                final_cost = to_goal_cost + speed_cost + ob_cost
                
                all_trajectories.append(trajectory)
                
                if final_cost < min_cost:
                    min_cost = final_cost
                    best_u = [tv, tw]
                    
        return best_u, min_cost, all_trajectories

    def calc_dynamic_window(self, v, w):
        Vs = [self.min_speed, self.max_speed,
              -self.max_yaw_rate, self.max_yaw_rate]
        
        Vd = [v - self.max_accel * self.dt,
              v + self.max_accel * self.dt,
              w - self.max_dyaw_rate * self.dt,
              w + self.max_dyaw_rate * self.dt]
        
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
        return dw

    def predict_trajectory(self, x, y, theta, v, w):
        traj = np.array([[x, y, theta, v, w]])
        time = 0
        while time <= self.predict_time:
            x += v * np.cos(theta) * self.dt
            y += v * np.sin(theta) * self.dt
            theta += w * self.dt
            time += self.dt
            traj = np.vstack((traj, [x, y, theta, v, w]))
        return traj

    def calc_to_goal_cost(self, traj, goal):
        dx = goal[0] - traj[-1, 0]
        dy = goal[1] - traj[-1, 1]
        error_angle = np.arctan2(dy, dx)
        cost_angle = error_angle - traj[-1, 2]
        cost = abs(np.arctan2(np.sin(cost_angle), np.cos(cost_angle)))
        return self.to_goal_cost_gain * cost

    def calc_obstacle_cost(self, traj, obstacles):
        if len(obstacles) == 0:
            return 0.0
            
        min_r = float('inf')
        
        ox = obstacles[:, 0]
        oy = obstacles[:, 1]
        
        for i in range(len(traj)):
            dx = traj[i, 0] - ox
            dy = traj[i, 1] - oy
            r = np.hypot(dx, dy)
            
            if np.min(r) <= self.robot_radius:
                return float('inf')
            
            if np.min(r) < min_r:
                min_r = np.min(r)
                
        return 1.0 / min_r * self.obstacle_cost_gain
