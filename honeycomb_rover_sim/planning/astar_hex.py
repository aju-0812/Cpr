import heapq
import math
from honeycomb_rover_sim.utils.geometry import HexUtils, normalize_angle

class AStarHex:
    def __init__(self, config):
        self.weights = config['planning']['weights']
        self.heuristic_scale = config['planning']['heuristic_scale']
        self.min_clearance = config['robot'].get('planning_min_clearance', config['robot']['radius'])

    def plan(self, grid, start_node, goal_node, start_heading=0.0):
        open_set = []
        # (f_score, g_score, current_node, current_heading)
        heapq.heappush(open_set, (0, 0, start_node, start_heading))
        
        came_from = {}
        g_score = {start_node: 0}
        
        while open_set:
            _, current_g, current, current_h = heapq.heappop(open_set)
            
            if current == goal_node:
                return self._reconstruct_path(came_from, current)
            
            for neighbor in grid.get_neighbors(current):
                if neighbor.clearance < self.min_clearance:
                    continue
                
                # Calculate edge cost
                dist = HexUtils.hex_distance(current.q, current.r, neighbor.q, neighbor.r) * grid.hex_size
                # Actually hex_distance returns steps. Multiply by hex_size * sqrt(3) for meters?
                # Let's use Euclidean for physical distance cost
                dist_m = math.hypot(neighbor.x - current.x, neighbor.y - current.y)
                
                # Heading cost
                target_heading = math.atan2(neighbor.y - current.y, neighbor.x - current.x)
                heading_diff = abs(normalize_angle(target_heading - current_h))
                
                # Cost function
                # cost = dist + alpha/(clearance+eps) + beta*importance + gamma*|d_heading|
                epsilon = 0.1
                clearance_cost = self.weights['alpha'] / (neighbor.clearance + epsilon)
                importance_cost = self.weights['beta'] * neighbor.importance
                heading_cost = self.weights['gamma'] * heading_diff
                
                edge_cost = dist_m + clearance_cost + importance_cost + heading_cost
                
                tentative_g = current_g + edge_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, target_heading))
                    
        return None # No path found

    def _heuristic(self, node, goal):
        return math.hypot(node.x - goal.x, node.y - goal.y) * self.heuristic_scale

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]
