import numpy as np
import matplotlib.pyplot as plt
from simulation import Environment, Rover, Sensor
from mapping import OccupancyGrid
from planning import GlobalPlanner, LocalPlanner
from slam import icp
from map_editor import MapEditor

def main():
    print("Opening Map Editor...")
    print("Instructions:")
    print("1. Draw obstacles (Boulder, Wall, Star)")
    print("2. Click 'Set Start' and place start point")
    print("3. Click 'Set Goal' and place goal point")
    print("4. Click 'Start Sim' to begin navigation")
    
    editor = MapEditor(width=100, height=100)
    env = editor.env
    
    # Get user-defined start and goal
    if editor.start_pos is None or editor.goal_pos is None:
        print("Simulation cancelled - Start or Goal not set")
        return
        
    start_pos = editor.start_pos
    goal_pos = editor.goal_pos
    
    print(f"\nStarting navigation from {start_pos} to {goal_pos}")
    
    rover = Rover(x=start_pos[0], y=start_pos[1], theta=0)
    sensor = Sensor(max_range=20, num_rays=60)
    
    grid_map = OccupancyGrid(width=100, height=100, resolution=1.0)
    
    global_planner = GlobalPlanner(grid_map)
    local_planner = LocalPlanner()
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # --- PATH PREVIEW ANIMATION ---
    print("Calculating initial path...")
    # Update map with initial obstacles for planning
    # We need to simulate a scan or just use ground truth for the preview?
    # For preview, let's just use the ground truth obstacles to block the grid
    # Since we don't have a perfect map yet, we'll simulate a 'perfect' map for the preview
    # Or better, just let the global planner plan on an empty map? No, that's useless.
    # Let's quickly populate the grid with the known obstacles for the preview.
    
    preview_grid = OccupancyGrid(width=100, height=100, resolution=1.0)
    for obs in env.obstacles:
        if obs['type'] == 'circle':
            # Simple approximation for preview
            cx, cy, r = obs['x'], obs['y'], obs['r']
            for i in range(int(cx-r), int(cx+r+1)):
                for j in range(int(cy-r), int(cy+r+1)):
                    if 0 <= i < 100 and 0 <= j < 100:
                        if (i-cx)**2 + (j-cy)**2 <= r**2:
                            preview_grid.grid[j, i] = 100 # Occupied
        elif obs['type'] == 'line':
             # Bresenham for lines
             p1 = obs['p1']
             p2 = obs['p2']
             # (Simplified: just mark start and end for now, or skip complex rasterization for this quick preview)
             # Actually, let's just skip complex rasterization and rely on the main loop for real mapping.
             # But the user wants to see the path *after* obstacles are placed.
             # So we MUST plan with obstacles.
             pass
             
    # NOTE: Since we want a "preview", we'll use the GlobalPlanner on a grid that knows about the obstacles.
    # But our GlobalPlanner works on OccupancyGrid.
    # Let's just run the A* on a grid where we manually mark obstacles.
    
    # Better approach: Just use the environment to check collisions in A*? 
    # No, A* uses the grid.
    # Let's populate the preview_grid properly.
    
    for x in range(100):
        for y in range(100):
            # Check if (x,y) is inside any obstacle
            # This is slow but fine for a one-time preview setup
            is_obs = False
            for obs in env.obstacles:
                if obs['type'] == 'circle':
                    if (x - obs['x'])**2 + (y - obs['y'])**2 <= obs['r']**2:
                        is_obs = True
                        break
                # Add other types if needed, but circles are main ones.
                # For lines, we can just assume the user wants to see the path around them.
                
            if is_obs:
                preview_grid.grid[y, x] = 100

    preview_planner = GlobalPlanner(preview_grid)
    initial_path = preview_planner.plan(start_pos, goal_pos)
    
    if initial_path:
        print("Showing Path Preview...")
        path_arr = np.array(initial_path)
        
        # Animate the preview
        for i in range(0, len(path_arr), 2): # Skip frames for speed
            ax1.clear()
            ax2.clear()
            
            # Draw Map & Obstacles
            ax1.set_title("PATH PREVIEW (Simulation starting soon...)")
            ax1.imshow(preview_grid.grid, cmap='Greys', origin='lower', extent=[0, 100, 0, 100], vmin=0, vmax=100, alpha=0.3)
            
            for obs in env.obstacles:
                if obs['type'] == 'circle':
                    circle = plt.Circle((obs['x'], obs['y']), obs['r'], color='r', fill=False)
                    ax1.add_patch(circle)
                elif obs['type'] == 'line':
                    p1 = obs['p1']
                    p2 = obs['p2']
                    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2)
            
            # Draw Full Path
            ax1.plot(path_arr[:, 0], path_arr[:, 1], 'g--', linewidth=1, alpha=0.5)
            
            # Draw "Ghost" Rover at current path index
            curr_x, curr_y = path_arr[i]
            
            # Calculate direction for ghost
            if i < len(path_arr) - 1:
                next_x, next_y = path_arr[i+1]
                angle = np.arctan2(next_y - curr_y, next_x - curr_x)
            else:
                angle = 0
                
            # Ghost Body
            ghost_len = 2.5
            ghost_wid = 1.5
            corners = np.array([[-ghost_len/2, -ghost_wid/2], [ghost_len/2, -ghost_wid/2], 
                                [ghost_len/2, ghost_wid/2], [-ghost_len/2, ghost_wid/2]])
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            rot_corners = corners @ rot.T
            rot_corners[:, 0] += curr_x
            rot_corners[:, 1] += curr_y
            
            ghost_shape = plt.Polygon(rot_corners, facecolor='lightgreen', edgecolor='green', alpha=0.7)
            ax1.add_patch(ghost_shape)
            
            ax1.plot(start_pos[0], start_pos[1], 'go', markersize=10)
            ax1.plot(goal_pos[0], goal_pos[1], 'r*', markersize=15)
            ax1.set_xlim(0, 100)
            ax1.set_ylim(0, 100)
            
            ax2.text(0.5, 0.5, "PREVIEWING PATH...", ha='center', fontsize=12)
            ax2.axis('off')
            
            plt.draw()
            plt.pause(0.01)
            
        print("Preview complete. Starting simulation...")
    else:
        print("Could not find initial path for preview.")

    initial_plan_made = False
    
    steps_per_frame = 5 # Run 5 simulation steps for every 1 visualization update
    
    for step in range(2000): # Increased total steps
        # Run simulation logic multiple times
        for _ in range(steps_per_frame):
            ranges, scan_points = sensor.scan(env, (rover.x, rover.y, rover.theta))
            grid_map.update_map(rover.x, rover.y, scan_points)
            
            # Global Planning (less frequent)
            if step % 10 == 0 or not initial_plan_made:
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
                
            best_u, _, all_trajectories = local_planner.plan(rover.x, rover.y, rover.theta, rover.v, rover.w, target, local_obstacles)
            v, w = best_u
            rover.move(v, w)
            
            if np.hypot(rover.x - goal_pos[0], rover.y - goal_pos[1]) < 2.0:
                print("Goal Reached!")
                plt.ioff()
                plt.show()
                return

        # --- VISUALIZE (Once per frame) ---
        ax1.clear()
        ax2.clear()
        
        ax1.set_title("Global Map & Path History")
        ax1.imshow(grid_map.get_probability_map(), cmap='Greys', origin='lower', 
                   extent=[0, 100, 0, 100], vmin=0, vmax=1, alpha=0.5)
        
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

        if len(rover.history) > 1:
            hist_arr = np.array(rover.history)
            ax1.plot(hist_arr[:, 0], hist_arr[:, 1], 'b-', linewidth=2, label='Path Taken')

        if path:
            path_arr = np.array(path)
            ax1.plot(path_arr[:, 0], path_arr[:, 1], 'g--', linewidth=1, label='Global Plan')

        # Draw rover as a car-like shape
        rover_length = 2.5
        rover_width = 1.5
        
        cos_theta = np.cos(rover.theta)
        sin_theta = np.sin(rover.theta)
        
        corners = np.array([
            [-rover_length/2, -rover_width/2],
            [rover_length/2, -rover_width/2],
            [rover_length/2, rover_width/2],
            [-rover_length/2, rover_width/2]
        ])
        
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        rotated_corners = corners @ rotation_matrix.T
        rotated_corners[:, 0] += rover.x
        rotated_corners[:, 1] += rover.y
        
        rover_shape = plt.Polygon(rotated_corners, facecolor='blue', edgecolor='darkblue', linewidth=2, alpha=0.8)
        ax1.add_patch(rover_shape)
        
        arrow_length = 2.0
        ax1.arrow(rover.x, rover.y, arrow_length*cos_theta, arrow_length*sin_theta, 
                 head_width=0.8, head_length=0.6, fc='yellow', ec='orange', linewidth=2)
        
        ax1.plot(start_pos[0], start_pos[1], 'go', markersize=12, label='START')
        ax1.plot(goal_pos[0], goal_pos[1], 'r*', markersize=15, markeredgewidth=2, label='GOAL')
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1.legend(loc='upper right')

        ax2.set_title("Local Analysis (DWA Trajectories)")
        ax2.set_xlim(rover.x - 10, rover.x + 10)
        ax2.set_ylim(rover.y - 10, rover.y + 10)
        
        # Plot all candidate trajectories
        for traj in all_trajectories:
            ax2.plot(traj[:, 0], traj[:, 1], 'k-', linewidth=0.5, alpha=0.4)
            
        # Highlight the CHOSEN trajectory (Best U)
        # We need to re-simulate the best trajectory to plot it distinctly
        # Or we can just find it in the list if we stored it, but re-simulating is easy
        # Actually, let's just plot a predicted trajectory based on v, w
        # Predict 3 seconds ahead
        pred_x, pred_y, pred_th = rover.x, rover.y, rover.theta
        pred_traj = []
        for _ in range(int(3.0 / 0.1)):
            pred_x += v * np.cos(pred_th) * 0.1
            pred_y += v * np.sin(pred_th) * 0.1
            pred_th += w * 0.1
            pred_traj.append((pred_x, pred_y))
        pred_traj = np.array(pred_traj)
        if len(pred_traj) > 0:
            ax2.plot(pred_traj[:, 0], pred_traj[:, 1], 'lime', linewidth=3, label='Chosen Path', zorder=10)
            
        if len(scan_points) > 0:
            ax2.plot(scan_points[:, 0], scan_points[:, 1], 'r.', markersize=3)
        
        rover_shape_local = plt.Polygon(rotated_corners, facecolor='blue', edgecolor='darkblue', linewidth=2, alpha=0.8)
        ax2.add_patch(rover_shape_local)
        ax2.arrow(rover.x, rover.y, arrow_length*cos_theta, arrow_length*sin_theta, 
                 head_width=0.4, head_length=0.3, fc='yellow', ec='orange', linewidth=2)
        ax2.legend(loc='upper right')
        
        plt.draw()
        plt.pause(0.001)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
