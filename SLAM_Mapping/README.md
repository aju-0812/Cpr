# 🗺️ Caterpillar-AI: SLAM & Autonomous Navigation

**Component:** Simultaneous Localization and Mapping (SLAM) & Path Planning  
**Status:** Prototype Complete  
**Developer:** Caterpillar-AI Team

---

## 📖 Project Overview
This module implements a complete **2D Navigation Stack** for the autonomous rover. It allows the rover to build a map of its environment, localize itself within that map, and autonomously plan paths to user-defined goals while avoiding obstacles.

The system combines **Global Planning** (A* Algorithm) for long-range route finding with **Local Planning** (Dynamic Window Approach - DWA) for reactive obstacle avoidance.

---

## ✨ Key Features

### 1. Interactive Map Editor
*   **Draw Your World**: Create custom environments by drawing obstacles (Circles, Lines, Polygons/Stars).
*   **Set Missions**: Interactively click to set the **Start Point** (Green) and **Goal Point** (Red).

### 2. Autonomous Navigation
*   **Global Planner (A*)**: Calculates the optimal path from Start to Goal through the known map.
*   **Local Planner (DWA)**: Controls the rover's velocity (`v`, `w`) to follow the global path while avoiding dynamic or unseen obstacles.
*   **Path Preview**: Animates a "Ghost Rover" to show the planned route *before* the simulation begins.

### 3. Advanced Visualization
*   **Dual-View Interface**:
    *   **Left Panel (Global Map)**: Shows the full map, ground truth obstacles, global path, and rover history.
    *   **Right Panel (Local Analysis)**: Shows what the rover "sees" (Lidar/Sensor data) and its decision-making process (Candidate Trajectories).
*   **Visual Rover**: The rover is rendered as a blue car-like agent with a direction arrow.
*   **Decision Lines**: A **Bright Green Line** shows exactly where the rover has decided to go next.

---

## 🚀 How to Run

### Prerequisites
```bash
pip install numpy matplotlib scipy
```

### Start the Simulation
```bash
python main.py
```

### Usage Instructions
1.  **Launch**: Run the script. The Map Editor window will open.
2.  **Edit Map**:
    *   Click **"Boulder"**, **"Wall"**, or **"Star"** to draw obstacles.
    *   Click **"Set Start"** and click on the map to place the rover.
    *   Click **"Set Goal"** and click on the map to place the destination.
3.  **Run**: Click **"Start Sim"**.
4.  **Watch**:
    *   First, watch the **Ghost Rover** preview the path.
    *   Then, watch the **Real Rover** navigate autonomously to the goal!

---

## 🧠 Technical Details

*   **SLAM**: Uses a grid-based occupancy map updated by simulated ray-casting sensors.
*   **A* (A-Star)**: Heuristic search algorithm for finding the shortest path on the grid.
*   **DWA (Dynamic Window Approach)**: Samples possible velocities and chooses the one that maximizes progress towards the goal while maintaining safety.

## 📂 File Structure

*   `main.py`: The entry point. Handles the GUI, simulation loop, and visualization.
*   `map_editor.py`: GUI for drawing maps and setting points.
*   `planning.py`: Contains the `GlobalPlanner` (A*) and `LocalPlanner` (DWA) classes.
*   `simulation.py`: Defines the `Rover`, `Sensor`, and `Environment` physics.
*   `slam.py`: Handles grid mapping and probability updates.
