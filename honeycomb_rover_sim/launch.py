import sys
import os

# Add the parent directory to sys.path to allow imports from honeycomb_rover_sim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeycomb_rover_sim.main import main

if __name__ == "__main__":
    main()
