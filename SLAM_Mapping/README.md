# SLAM and Navigation Prototype

Software-only rover navigation system with **user-defined start and goal points**. Features A* global planning, DWA local planning, and real-time dual-view visualization.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Usage

1. **Draw Obstacles**: Use Boulder, Wall, or Star buttons
2. **Set Start**: Click "Set Start" button, then click on map
3. **Set Goal**: Click "Set Goal" button, then click on map
4. **Start Simulation**: Click "Start Sim" to watch autonomous navigation

## Features

- **Interactive Map Editor**: Full control over environment setup
- **User-Defined Start/Goal**: Place start and goal anywhere
- **A* Global Planning**: Optimal path finding
- **DWA Local Planning**: Smooth obstacle avoidance (3x speed)
- **Dual-View Visualization**: 
  - Left: Global map with path history
  - Right: Local analysis showing DWA trajectories

## Testing

```bash
python test_headless.py
```
