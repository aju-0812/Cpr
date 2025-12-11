import numpy as np
import matplotlib.pyplot as plt
from simulation import Environment, Rover, Sensor
from mapping import OccupancyGrid
from planning import GlobalPlanner, LocalPlanner
from slam import icp
from map_editor import MapEditor

def main():
    print("Opening Map Editor...")
    editor = MapEditor(width=100, height=100)
    env = editor.env
    
    # Check if map is empty, add default obstacles if so
    if not env.obstacles:
        print("No obstacles drawn. Adding default obstacles.")
        env.add_obstacle(30, 30, 5)
        env.add_obstacle(60, 40, 8)
        env.add_obstacle(20, 70, 6)
        env.add_obstacle(70, 70, 5)
        env.add_rectangle_obstacle(40, 50, 10, 20)
    
    # Define Start and Goal
    start_pos = (5, 5)
    goal_pos = (95, 95)
    
    rover = Rover(x=start_pos[0], y=start_pos[1], theta=0)
    sensor = Sensor(max_range=20, num_rays=60)
    
    grid_map = OccupancyGrid(width=100, height=100, resolution=1.0)
    
    global_planner = GlobalPlanner(grid_map)
    local_planner = LocalPlanner()
    
    plt.ion()
    # Dual View: Left = Global, Right = Local Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Initial Plan Flag
    initial_plan_made = False
    
    for step in range(1000):
        ranges, scan_points = sensor.scan(env, (rover.x, rover.y, rover.theta))
        
        grid_map.update_map(rover.x, rover.y, scan_points)
        
        # --- PLAN ---
        # Replan global path frequently (every 2 steps) for faster analysis
        if step % 2 == 0 or not initial_plan_made:
            path = global_planner.plan((rover.x, rover.y), goal_pos)
            if path:
                initial_plan_made = True
        
        local_obstacles = np.array(scan_points) if len(scan_points) > 0 else np.empty((0, 2))
        
        if path and len(path) > 1:
            lookahead_dist = 4.0
            target_idx = 0
            for i, p in enumerate(path):
                if np.hypot(p[0] - rover.x, p[1] - rover.y) > lookahead_dist:
                    target_idx = i
                    break
            target = path[target_idx]
        else:
            target = goal_pos
            
        # Get best control AND all candidate trajectories
        best_u, _, all_trajectories = local_planner.plan(rover.x, rover.y, rover.theta, rover.v, rover.w, target, local_obstacles)
        v, w = best_u
        
        rover.move(v, w)
        
        # --- VISUALIZE ---
        if step % 2 == 0: # Update viz every 2 steps for speed
            ax1.clear()
            ax2.clear()
            
            # --- LEFT PANEL: GLOBAL MAP ---
            ax1.set_title("Global Map & Path History")
            ax1.imshow(grid_map.get_probability_map(), cmap='Greys', origin='lower', 
                       extent=[0, 100, 0, 100], vmin=0, vmax=1, alpha=0.5)
            
            # Ground Truth Obstacles (Reference)
            for obs in env.obstacles:
                if obs['type'] == 'circle':
                    circle = plt.Circle((obs['x'], obs['y']), obs['r'], color='r', fill=False, alpha=0.3)
                    ax1.add_patch(circle)
                elif obs['type'] == 'rect':
                    rect = plt.Rectangle((obs['x'], obs['y']), obs['w'], obs['h'], color='r', fill=False, alpha=0.3)
                    ax1.add_patch(rect)
                elif obs['type'] == 'line':
                    p1 = obs['p1']
                    p2 = obs['p2']
                    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=1, alpha=0.3)
                elif obs['type'] == 'polygon':
                    poly = plt.Polygon(obs['points'], edgecolor='r', facecolor='none', alpha=0.3)
                    ax1.add_patch(poly)

            # Rover History (Actual Path Taken) - Solid Blue Line
            if len(rover.history) > 1:
                hist_arr = np.array(rover.history)
                ax1.plot(hist_arr[:, 0], hist_arr[:, 1], 'b-', linewidth=2, label='Path Taken')

            # Global Plan - Dashed Green Line (Future)
            if path:
                path_arr = np.array(path)
                ax1.plot(path_arr[:, 0], path_arr[:, 1], 'g--', linewidth=1, label='Global Plan')

            ax1.plot(rover.x, rover.y, 'bo', markersize=8)
            ax1.plot(goal_pos[0], goal_pos[1], 'gx', markersize=10, markeredgewidth=2)
            ax1.set_xlim(0, 100)
            ax1.set_ylim(0, 100)
            ax1.legend(loc='upper right')

            # --- RIGHT PANEL: LOCAL ANALYSIS ---
            ax2.set_title("Local Analysis (DWA Trajectories)")
            ax2.set_xlim(rover.x - 10, rover.x + 10)
            ax2.set_ylim(rover.y - 10, rover.y + 10)
            
            # Plot all candidate trajectories (Thin grey lines)
            for traj in all_trajectories:
                ax2.plot(traj[:, 0], traj[:, 1], 'k-', linewidth=0.5, alpha=0.2)
                
            # Plot Scan Points (Red Dots)
            if len(scan_points) > 0:
                ax2.plot(scan_points[:, 0], scan_points[:, 1], 'r.', markersize=3)
                
            # Plot Rover
            ax2.plot(rover.x, rover.y, 'bo', markersize=10)
            ax2.arrow(rover.x, rover.y, 2*np.cos(rover.theta), 2*np.sin(rover.theta), head_width=0.5, color='b')
            
            plt.draw()
            plt.pause(0.001) # Faster pause
            
        if np.hypot(rover.x - goal_pos[0], rover.y - goal_pos[1]) < 2.0:
            print("Goal Reached!")
            break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
