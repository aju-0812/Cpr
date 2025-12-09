**AprilTag Sequential Navigation using OpenCV (Python)**

This project implements a sequential navigation system using AprilTags and a normal webcam (OpenCV).
The program detects AprilTags in a specific order, calculates the rover’s direction (Left / Right / Straight), and marks Mission Complete after all tags are detected in sequence.

**Features**
**AprilTag Detection:**

Uses pupil_apriltags for fast and accurate AprilTag detection.

**Sequential Navigation Logic:**

Detects tags in order:

2 → 3 → 1

Moves to the next stage only when the correct tag is detected.


Direction Estimation:
Based on the tag’s X-position in the frame:

TURN LEFT
TURN RIGHT
GO STRAIGHT


Single Window Output
Displays:

Live camera feed
AprilTag ID and bounding box
Navigation direction
Current target tag
Mission completion message
