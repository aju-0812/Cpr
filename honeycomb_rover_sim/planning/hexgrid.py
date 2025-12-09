import numpy as np
from dataclasses import dataclass
from honeycomb_rover_sim.utils.geometry import HexUtils

@dataclass
class HexNode:
    q: int
    r: int
    x: float
    y: float
    clearance: float
    zone: str
    importance: float = 0.0
    walkable: bool = True

    def __eq__(self, other):
        return isinstance(other, HexNode) and self.q == other.q and self.r == other.r

    def __hash__(self):
        return hash((self.q, self.r))

class HexGrid:
    def __init__(self, arena, hex_size):
        self.arena = arena
        self.hex_size = hex_size
        self.nodes = {} # (q, r) -> HexNode

        self._generate_grid()

    def _generate_grid(self):
        # Determine bounds in q, r
        # This is a bit brute force: iterate over pixel space and convert
        # Better: calculate q,r bounds
        
        # Estimate range
        max_dim = max(self.arena.width, self.arena.height)
        range_limit = int(max_dim / self.hex_size * 2) # Safety factor

        for q in range(-range_limit, range_limit):
            for r in range(-range_limit, range_limit):
                x, y = HexUtils.hex_to_pixel(q, r, self.hex_size)
                
                # Check if center is inside arena
                if 0 <= x <= self.arena.width and 0 <= y <= self.arena.height:
                    clearance = self.arena.get_clearance(x, y)
                    zone = self.arena.get_zone_at(x, y)
                    
                    # Walkable if clearance > 0 (will refine with robot radius later)
                    # Actually, let's mark walkable if inside bounds.
                    # A* will check clearance threshold.
                    
                    self.nodes[(q, r)] = HexNode(
                        q=q, r=r, x=x, y=y,
                        clearance=clearance,
                        zone=zone
                    )

    def get_node(self, q, r):
        return self.nodes.get((q, r))

    def get_neighbors(self, node):
        neighbors = []
        for nq, nr in HexUtils.get_neighbors(node.q, node.r):
            if (nq, nr) in self.nodes:
                neighbors.append(self.nodes[(nq, nr)])
        return neighbors
