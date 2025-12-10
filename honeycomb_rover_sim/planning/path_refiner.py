import numpy as np
from scipy.interpolate import splprep, splev

class PathRefiner:
    def __init__(self, config):
        self.ds = 0.1 # Resample resolution
        self.min_turn_radius = 1.5 # From spec: max curvature < 1/1.5m
        self.max_v = config['robot']['max_v']

    def refine(self, hex_path):
        if not hex_path or len(hex_path) < 2:
            return None
            
        # Extract points
        x = [n.x for n in hex_path]
        y = [n.y for n in hex_path]
        
        # Add duplicates at ends to clamp spline if needed, but standard should be fine
        # If path is short (2 points), simple linear interpolation
        if len(hex_path) <= 3:
            k = 1
        else:
            k = 3
            
        try:
            tck, u = splprep([x, y], s=0.1, k=k) # s is smoothing factor
        except Exception as e:
            print(f"Spline failed: {e}")
            return None

        # Resample
        # Estimate length to determine number of points
        approx_len = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        num_points = int(approx_len / self.ds)
        u_new = np.linspace(0, 1, num_points)
        
        x_new, y_new = splev(u_new, tck)
        
        # Calculate derivatives for heading and curvature
        dx, dy = splev(u_new, tck, der=1)
        ddx, ddy = splev(u_new, tck, der=2)
        
        headings = np.arctan2(dy, dx)
        
        # Curvature k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        curvature = np.abs(dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
        
        # Velocity profile
        # Slow down for high curvature or low clearance (not checking clearance here for simplicity, 
        # but could pass grid if needed)
        velocities = []
        for k in curvature:
            # v^2 / r < a_lat_max? Or just simple scaling
            # Limit v based on curvature: v < sqrt(a_max * r) = sqrt(a_max / k)
            # Let's just linearly scale down
            if k > 1.0 / self.min_turn_radius:
                v = self.max_v * 0.5 # Slow down significantly
            else:
                v = self.max_v
            velocities.append(v)
            
        path_points = []
        for i in range(len(x_new)):
            path_points.append({
                'x': x_new[i],
                'y': y_new[i],
                'yaw': headings[i],
                'v': velocities[i],
                'k': curvature[i]
            })
            
        return path_points
