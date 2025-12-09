import sys
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from honeycomb_rover_sim.utils.image_parser import parse_arena_image
from honeycomb_rover_sim.world.arena import Arena
from honeycomb_rover_sim.world.markers import MarkerMap
from honeycomb_rover_sim.planning.hexgrid import HexGrid
from honeycomb_rover_sim.planning.importance import ImportanceEngine
from honeycomb_rover_sim.planning.astar_hex import AStarHex
from honeycomb_rover_sim.planning.path_refiner import PathRefiner
from honeycomb_rover_sim.utils.geometry import HexUtils

def run_demo(image_path, output_path):
    print(f"Processing {image_path}...")
    
    # Read Image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        
    # Parse
    map_config = parse_arena_image(image_bytes)
    print("Map parsed successfully.")
    print(f"Obstacles: {len(map_config['obstacles'])}")
    print(f"Zones: {map_config['zones'].keys()}")
    if 'start' in map_config['zones']:
        print(f"Start Zone: {map_config['zones']['start']['rect']}")
    print("Obstacles:")
    for o in map_config['obstacles']:
        print(f"  - ({o['x']:.2f}, {o['y']:.2f}, r={o['r']:.2f})")
    print(f"Markers: {len(map_config['markers'])}")
    if map_config['markers']:
        print(f"Goal Marker: {map_config['markers'][0]}")
    
    # Load Config
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config', 'params.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Setup
    arena = Arena(map_config)
    marker_map = MarkerMap(map_config)
    hex_grid = HexGrid(arena, config['grid']['hex_size'])
    importance_engine = ImportanceEngine(config)
    
    # Start/Goal
    start_pose = config['robot']['start_pose']
    start_q, start_r = HexUtils.pixel_to_hex_round(start_pose[0], start_pose[1], config['grid']['hex_size'])
    start_node = hex_grid.get_node(start_q, start_r)
    
    goal_marker = marker_map.get_marker(0)
    if not goal_marker:
        print("No goal marker found in image!")
        return
        
    goal_q, goal_r = HexUtils.pixel_to_hex_round(goal_marker.x, goal_marker.y, config['grid']['hex_size'])
    goal_node = hex_grid.get_node(goal_q, goal_r)
    
    if not start_node or not goal_node:
        print("Start or Goal outside grid.")
        return
        
    # Plan
    importance_engine.compute_importance(hex_grid, goal_node)
    planner = AStarHex(config)
    path = planner.plan(hex_grid, start_node, goal_node, start_heading=start_pose[2])
    
    if path:
        print(f"Path found: {len(path)} steps.")
        refiner = PathRefiner(config)
        refined_path = refiner.refine(path)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_aspect('equal')
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 5)
        
        # Grid
        nodes = list(hex_grid.nodes.values())
        x = [n.x for n in nodes]
        y = [n.y for n in nodes]
        c = [n.importance for n in nodes]
        ax.scatter(x, y, c=c, cmap='coolwarm', s=10, alpha=0.5)
        
        # Obstacles
        for obs in arena.obstacles:
            circle = plt.Circle((obs.x, obs.y), obs.r, color='black', alpha=0.7)
            ax.add_patch(circle)
            
        # Path
        px = [n.x for n in path]
        py = [n.y for n in path]
        ax.plot(px, py, 'g--', linewidth=2, label='A*')
        
        if refined_path:
            rpx = [p['x'] for p in refined_path]
            rpy = [p['y'] for p in refined_path]
            ax.plot(rpx, rpy, 'c-', linewidth=2, label='Refined')
            
        ax.legend()
        plt.title("Pathfinding Result")
        plt.savefig(output_path)
        print(f"Result saved to {output_path}")
    else:
        print("No path found.")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        run_demo(sys.argv[1], sys.argv[2])
