import numpy as np
from simulation import Environment, Rover, Sensor
from mapping import OccupancyGrid
from planning import GlobalPlanner, LocalPlanner

def test_headless():
    print("Starting headless test...")
    try:
        env = Environment(width=100, height=100)
        env.add_obstacle(30, 30, 5)
        
        rover = Rover(x=10, y=10, theta=0)
        sensor = Sensor(max_range=20, num_rays=10)
        grid_map = OccupancyGrid(width=100, height=100, resolution=1.0)
        global_planner = GlobalPlanner(grid_map)
        local_planner = LocalPlanner()
        
        goal = (80, 80)
        
        for step in range(50):
            ranges, scan_points = sensor.scan(env, (rover.x, rover.y, rover.theta))
            grid_map.update_map(rover.x, rover.y, scan_points)
            
            if step % 10 == 0:
                path = global_planner.plan((rover.x, rover.y), goal)
                
            local_obstacles = np.array(scan_points) if len(scan_points) > 0 else np.empty((0, 2))
            best_u, _, _ = local_planner.plan(rover.x, rover.y, rover.theta, rover.v, rover.w, goal, local_obstacles)
            rover.move(best_u[0], best_u[1])
            
        print("Headless test passed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise e

if __name__ == "__main__":
    test_headless()
