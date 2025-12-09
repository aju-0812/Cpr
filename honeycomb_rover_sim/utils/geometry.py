import numpy as np
import math

class HexUtils:
    """
    Utilities for Pointy-topped Hexagonal Grid (Axial Coordinates q, r).
    """

    @staticmethod
    def hex_to_pixel(q, r, size):
        """Convert axial (q, r) to pixel (x, y) center."""
        x = size * (math.sqrt(3) * q + math.sqrt(3)/2 * r)
        y = size * (3./2 * r)
        return x, y

    @staticmethod
    def pixel_to_hex(x, y, size):
        """Convert pixel (x, y) to fractional axial (q, r)."""
        q = (math.sqrt(3)/3 * x - 1./3 * y) / size
        r = (2./3 * y) / size
        return q, r

    @staticmethod
    def hex_round(q, r):
        """Round fractional axial coordinates to nearest integer hex."""
        s = -q - r
        rq, rr, rs = round(q), round(r), round(s)
        q_diff = abs(rq - q)
        r_diff = abs(rr - r)
        s_diff = abs(rs - s)

        if q_diff > r_diff and q_diff > s_diff:
            rq = -rr - rs
        elif r_diff > s_diff:
            rr = -rq - rs
        # else s changes, but we don't store s
        return int(rq), int(rr)

    @staticmethod
    def pixel_to_hex_round(x, y, size):
        q, r = HexUtils.pixel_to_hex(x, y, size)
        return HexUtils.hex_round(q, r)

    @staticmethod
    def hex_distance(q1, r1, q2, r2):
        """Manhattan distance on hex grid."""
        return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) / 2

    @staticmethod
    def get_neighbors(q, r):
        """Get 6 neighbors of a hex."""
        directions = [
            (1, 0), (1, -1), (0, -1),
            (-1, 0), (-1, 1), (0, 1)
        ]
        return [(q + dq, r + dr) for dq, dr in directions]

    @staticmethod
    def get_hex_corners(x, y, size):
        """Get the 6 corners of a hex centered at x, y."""
        corners = []
        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = math.pi / 180 * angle_deg
            corners.append((x + size * math.cos(angle_rad),
                            y + size * math.sin(angle_rad)))
        return corners

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
