**AprilTag Sequential Navigation using OpenCV (Python)**

This project implements a sequential navigation system using AprilTags and a normal webcam (OpenCV).
The program detects AprilTags in a specific order, calculates the rover’s direction (Left / Right / Straight), and marks Mission Complete after all tags are detected in sequence.

**Features**
**AprilTag Detection**

Uses pupil_apriltags for fast and accurate AprilTag detection.

**Sequential Navigation Logic**

Detects tags in order:

2 → 3 → 1


Moves to the next stage only when the correct tag is detected.

**Direction Estimation**

Based on the tag’s X-position in the frame:

TURN LEFT

TURN RIGHT

GO STRAIGHT

**Single Window Output**

**Displays:**

Live camera feed

AprilTag ID and bounding box

Navigation direction

Current target tag

Mission completion message

--------------------------------------------------------------------------------------------------------------------------------------------------------------------



**YOLOv8 Object Detection with Depth Estimation (OAK-D Lite)**

This project integrates YOLOv8 object detection with stereo depth estimation using the Luxonis OAK-D Lite camera.
The script performs real-time detection, calculates object distance from the depth map, and visualizes results on both RGB and depth windows.

**Features**

Real-time YOLOv8 object detection

Stereo depth map generation using OAK-D Lite

Object distance estimation (in meters) using the center depth pixel

Bounding boxes displayed on both RGB and Depth windows

Colorized depth map for better visualization

Requirements

**Install dependencies:**

pip install depthai ultralytics opencv-python numpy


**Download a YOLO model (example uses yolov8s):**

from ultralytics import YOLO
model = YOLO("yolov8s.pt")


**Hardware Required:**

OAK-D Lite or DepthAI stereo camera


**How It Works**

YOLOv8 runs on RGB preview frames.

StereoDepth node computes depth in millimeters.

The object’s center pixel is used to estimate distance.

**RGB view shows:**

Bounding box

Object name + distance

**Depth map shows:**

Colorized depth

Bounding box

Center point + distance
