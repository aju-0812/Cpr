# Honeycomb Rover Simulation

A modular, pure Python simulation of an autonomous rover navigating a hexagonal grid arena.

## Features
- **Hexagonal Grid**: Pointy-top axial coordinate system.
- **Importance Engine**: Potential field-based cost map (clearance, zones, goal attraction).
- **A* Pathfinding**: Custom A* on hex grid with heading and clearance costs.
- **Path Refinement**: B-spline smoothing and velocity profiling.
- **Local Planner**: Pure Pursuit controller with simple obstacle avoidance.
- **Visualization**: Live Matplotlib animation.

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Standard Simulation
Run the default simulation with the hardcoded map:
```bash
python honeycomb_rover_sim/launch.py
```

### 2. Streamlit Planner (Web Interface)
Run the web interface to upload your own map and save models:
```bash
streamlit run implementation/streamlit_app/app.py
```
1. Upload an image of the arena.
2. Wait for path generation.
3. Click **"Save Model"** to save the grid state to `implementation/model/grid_model.joblib`.

## Configuration
- `config/params.yaml`: Tune simulation, grid, and robot parameters.
- `config/map_config.yaml`: Edit arena layout, obstacles, and markers.

## Structure
- `world/`: Arena and marker definitions.
- `robot/`: Robot kinematics and controller.
- `planning/`: Pathfinding and mission logic.
- `perception/`: Sensor simulation.
- `utils/`: Geometry and visualization tools.
