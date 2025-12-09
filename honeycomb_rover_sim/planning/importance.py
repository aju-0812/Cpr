import numpy as np
from scipy.ndimage import gaussian_filter

class ImportanceEngine:
    def __init__(self, config):
        self.cfg = config['importance']
        self.smoothing_sigma = config['planning']['smoothing']['sigma']

    def compute_importance(self, grid, goal_node):
        """
        Compute importance score for all nodes in the grid.
        I = lambda1 * (1 - clearance_norm) + lambda2 * zone_penalty + lambda3 * marker_attraction
        """
        nodes = list(grid.nodes.values())
        if not nodes:
            return

        # Extract values for vectorized computation
        clearances = np.array([n.clearance for n in nodes])
        max_clearance = np.max(clearances) if np.max(clearances) > 0 else 1.0
        norm_clearance = np.clip(clearances / max_clearance, 0, 1)

        # Zone penalties
        zone_scores = np.zeros(len(nodes))
        for i, n in enumerate(nodes):
            if n.zone == 'danger': # Excavation
                zone_scores[i] = self.cfg['excavation_penalty']
            elif n.zone == 'safe': # Construction
                zone_scores[i] = self.cfg['construction_bonus']
        
        # Goal attraction (negative score)
        # Using normalized distance to goal
        dists = np.array([np.hypot(n.x - goal_node.x, n.y - goal_node.y) for n in nodes])
        max_dist = np.max(dists) if np.max(dists) > 0 else 1.0
        attraction = -(1.0 - (dists / max_dist)) # Closer is more negative (better)
        
        # Combine
        # Note: The prompt formula said:
        # Near goal marker -> strong negative (attractive)
        # I = lambda1*(1-clearance_norm) + lambda2*zone_penalty + lambda3*marker_attraction
        # marker_attraction should be negative for close, 0 for far? 
        # Or maybe the formula implies attraction is a term that reduces cost.
        # Let's assume attraction term is NEGATIVE for good spots.
        
        raw_scores = (self.cfg['lambda1'] * (1 - norm_clearance) +
                      self.cfg['lambda2'] * zone_scores +
                      self.cfg['lambda3'] * attraction)

        # Apply to nodes temporarily
        q_vals = [n.q for n in nodes]
        r_vals = [n.r for n in nodes]
        min_q, max_q = min(q_vals), max(q_vals)
        min_r, max_r = min(r_vals), max(r_vals)
        
        # Create 2D array for smoothing (mapping axial to array indices)
        # Shift to 0-index
        w = max_q - min_q + 1
        h = max_r - min_r + 1
        score_map = np.zeros((w, h))
        mask_map = np.zeros((w, h))

        for i, n in enumerate(nodes):
            qi = n.q - min_q
            ri = n.r - min_r
            score_map[qi, ri] = raw_scores[i]
            mask_map[qi, ri] = 1

        # Smooth
        # Note: standard gaussian filter on axial grid is an approximation but acceptable
        smoothed_map = gaussian_filter(score_map, sigma=self.smoothing_sigma)

        # Assign back
        for n in nodes:
            qi = n.q - min_q
            ri = n.r - min_r
            if mask_map[qi, ri]:
                n.importance = smoothed_map[qi, ri]
                # Clamp to [-1, 1] as per spec
                n.importance = max(-1.0, min(1.0, n.importance))
