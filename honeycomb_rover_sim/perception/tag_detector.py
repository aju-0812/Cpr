import numpy as np
from honeycomb_rover_sim.utils.geometry import normalize_angle

class TagDetector:
    def __init__(self, fov=120, max_range=5.0):
        self.fov = np.deg2rad(fov)
        self.max_range = max_range

    def detect(self, robot_pose, markers):
        """
        Returns list of detected markers.
        robot_pose: (x, y, yaw)
        markers: list of Marker objects
        """
        rx, ry, ryaw = robot_pose
        detected = []

        for m in markers:
            dist = np.hypot(m.x - rx, m.y - ry)
            if dist > self.max_range:
                continue

            angle_to_marker = np.arctan2(m.y - ry, m.x - rx)
            angle_diff = normalize_angle(angle_to_marker - ryaw)

            if abs(angle_diff) <= self.fov / 2.0:
                detected.append({
                    'id': m.id,
                    'range': dist,
                    'bearing': angle_diff,
                    'type': m.type
                })
        
        return detected
