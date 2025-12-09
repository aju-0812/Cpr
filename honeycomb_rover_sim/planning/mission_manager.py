from enum import Enum, auto

class MissionState(Enum):
    PLANNING = auto()
    EXECUTING = auto()
    APPROACHING = auto()
    SCANNING = auto()
    REPORTING = auto()
    SUCCESS = auto()
    FAILURE = auto()

class MissionManager:
    def __init__(self):
        self.state = MissionState.PLANNING
        self.target_marker_id = 0
        self.detected_markers = []
        self.start_time = 0
        self.mission_log = []

    def update(self, robot, dist_to_goal):
        if self.state == MissionState.EXECUTING:
            if dist_to_goal < 1.0:
                self.state = MissionState.APPROACHING
                self.log("Entering approach phase")
        
        elif self.state == MissionState.APPROACHING:
            if dist_to_goal < 0.5:
                self.state = MissionState.SCANNING
                self.log("Goal reached. Starting scan.")
                return "SCAN" # Signal to robot to spin
        
        elif self.state == MissionState.SCANNING:
            # Logic handled by main loop to check if scan complete
            pass

    def log(self, msg):
        self.mission_log.append(msg)
        print(f"[MISSION] {msg}")
