# SLAM and Navigation Prototype

This project implements a software-only prototype for a rover's mapping and navigation system. It simulates a rover with a depth sensor, builds an occupancy grid map, and navigates using Global (A*) and Local (DWA) planners.

## Workflow

1.  **Simulation (`simulation.py`)**:
    -   Creates a virtual environment with obstacles.
    -   Simulates a differential drive rover.
    -   Simulates a LiDAR/Depth sensor using raycasting.

2.  **Mapping (`mapping.py`)**:
    -   Takes sensor scans and rover pose.
    -   Updates a 2D Occupancy Grid using log-odds probabilities.
    -   Uses Bresenham's algorithm to clear free space along sensor rays.

3.  **SLAM (`slam.py`)**:
    -   Provides the ICP (Iterative Closest Point) algorithm for scan matching and alignment (used for odometry correction).

4.  **Planning (`planning.py`)**:
    -   **Global Planner**: Uses A* to find a path from the rover to the goal on the occupancy grid.
    -   **Local Planner**: Uses Dynamic Window Approach (DWA) to generate velocity commands that follow the global path while avoiding local obstacles.

5.  **Interactive Map Editor (`map_editor.py`)**:
    -   Allows users to draw the environment before simulation starts.
    -   Features buttons for easy mode selection (Boulder, Wall, Star).

6.  **Integration (`main.py`)**:
    -   Runs the main loop: Sense -> Map -> Plan -> Move.
    -   **Dual-View Visualization**: Shows Global Map on the left and Local Analysis (DWA trajectories) on the right.

## How to Run

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the simulation:
    ```bash
    python main.py
    ```
    -   **Draw your map**: Use the buttons at the bottom to select obstacles and place them on the map.
    -   **Start**: Click "Start Sim" to begin.
    -   **Observe**: Watch the rover navigate in the split-screen view.

3.  Run headless test (verification):
    ```bash
    python test_headless.py
    ```
