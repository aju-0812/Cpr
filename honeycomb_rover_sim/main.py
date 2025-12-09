import yaml
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from honeycomb_rover_sim.world.arena import Arena
from honeycomb_rover_sim.world.markers import MarkerMap
from honeycomb_rover_sim.robot.robot import Robot
from honeycomb_rover_sim.robot.controller import Controller
from honeycomb_rover_sim.planning.hexgrid import HexGrid
from honeycomb_rover_sim.planning.importance import ImportanceEngine
from honeycomb_rover_sim.planning.astar_hex import AStarHex
from honeycomb_rover_sim.planning.path_refiner import PathRefiner
from honeycomb_rover_sim.planning.mission_manager import MissionManager, MissionState
from honeycomb_rover_sim.perception.tag_detector import TagDetector
from honeycomb_rover_sim.utils.visualization import Visualizer
from honeycomb_rover_sim.utils.geometry import HexUtils
from honeycomb_rover_sim.utils.logger import logger

def main():
    # Load Config
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config', 'params.yaml')
    map_config_path = os.path.join(base_dir, 'config', 'map_config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize World
    arena = Arena(map_config_path)
    marker_map = MarkerMap(map_config_path)
    
    # Initialize Planning
    logger.info("Generating Hex Grid...")
    hex_grid = HexGrid(arena, config['grid']['hex_size'])
    logger.info(f"Grid generated with {len(hex_grid.nodes)} nodes.")
    
    importance_engine = ImportanceEngine(config)
    
    # Identify Start and Goal Nodes
    # Start: Robot start pose
    start_pose = config['robot']['start_pose']
    start_q, start_r = HexUtils.pixel_to_hex_round(start_pose[0], start_pose[1], config['grid']['hex_size'])
    start_node = hex_grid.get_node(start_q, start_r)
    
    # Goal: Marker 0
    goal_marker = marker_map.get_marker(0)
    goal_q, goal_r = HexUtils.pixel_to_hex_round(goal_marker.x, goal_marker.y, config['grid']['hex_size'])
    goal_node = hex_grid.get_node(goal_q, goal_r)
    
    if not start_node or not goal_node:
        logger.error(f"Start or Goal node outside grid! Start: {start_node}, Goal: {goal_node}")
        return

    logger.info(f"Start Node: q={start_node.q}, r={start_node.r}, clearance={start_node.clearance:.2f}")
    logger.info(f"Goal Node: q={goal_node.q}, r={goal_node.r}, clearance={goal_node.clearance:.2f}")

    # Compute Importance
    logger.info("Computing Importance Field...")
    importance_engine.compute_importance(hex_grid, goal_node)
    
    # Plan Path
    logger.info("Planning Path...")
    planner = AStarHex(config)
    path = planner.plan(hex_grid, start_node, goal_node, start_heading=start_pose[2])
    
    if not path:
        logger.error("No path found!")
        return
        
    logger.info(f"Path found with {len(path)} nodes.")
    
    # Refine Path
    refiner = PathRefiner(config)
    refined_path = refiner.refine(path)
    
    # Initialize Robot & Control
    robot = Robot(config)
    controller = Controller(config)
    mission_manager = MissionManager()
    mission_manager.state = MissionState.EXECUTING
    detector = TagDetector()
    
    # Visualization
    viz = Visualizer(arena, hex_grid, robot, config)
    
    # Simulation Loop
    dt = config['sim']['dt']
    max_time = config['sim']['max_time']
    steps = int(max_time / dt)
    
    def update(frame):
        nonlocal robot, mission_manager
        
        if mission_manager.state == MissionState.SUCCESS:
            return
            
        # 1. Perception
        detections = detector.detect(robot.get_pose(), marker_map.markers)
        
        # 2. Mission Update
        dist_to_goal = np.hypot(robot.x - goal_marker.x, robot.y - goal_marker.y)
        action = mission_manager.update(robot, dist_to_goal)
        
        # 3. Control
        v, w = 0.0, 0.0
        if action == "SCAN":
            # Spin
            v = 0.0
            w = 0.5
            if len(detections) > 0:
                logger.info(f"Scanned markers: {[d['id'] for d in detections]}")
                mission_manager.state = MissionState.SUCCESS
        elif mission_manager.state != MissionState.SUCCESS:
            v, w = controller.compute_command(robot, refined_path, arena)
            
        # 4. Robot Update
        robot.update(v, w, dt)
        
        # 5. Viz Update
        return viz.update(robot, path, refined_path, detections)
        
    anim = FuncAnimation(viz.fig, update, frames=steps, interval=20, blit=True, repeat=False)
    plt.show()

if __name__ == "__main__":
    main()
