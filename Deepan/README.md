**AprilTag-Based Sequential Navigation System
Overview**

This project implements a vision-based sequential navigation system using AprilTags and OpenCV.
A camera detects AprilTags in real time and guides a rover (or robot) through predefined navigation zones by following a fixed tag order.

The system determines whether the rover should turn left, turn right, or move straight based on the horizontal position of the detected AprilTag in the camera frame.

**Key Features**

✅ Real-time AprilTag detection

✅ Sequential navigation using predefined tag IDs

✅ Direction guidance (LEFT / RIGHT / STRAIGHT)

✅ Visual overlay of tag boundaries, center point, and commands

✅ Single-window display with mission status

✅ Easily extendable to motor control or ROS integration

**Navigation Logic**

The rover must detect AprilTags in a fixed sequence:

NAV_SEQUENCE = [2, 3, 1]


The system searches only for the current target tag

Once detected, it:

Computes the tag’s center position

Decides movement direction

Advances to the next navigation stage

After all tags are detected → Mission Complete

**Direction Decision Rule**


The camera frame width is divided into three regions:

Tag Position	Rover Action
Left of center	TURN LEFT
Right of center	TURN RIGHT
Near center	GO STRAIGHT

Thresholds are defined as:

LEFT_THRESHOLD  = CENTER_X - 40
RIGHT_THRESHOLD = CENTER_X + 40

**Visual Output**

Each detected tag displays:

Green bounding box

Red center point

Tag ID label

Movement direction

Mission status text

All information is shown in one single display window.

**Requirements**

Install the required Python libraries:

pip install opencv-python numpy pupil-apriltags

Hardware

USB Camera / Laptop Camera

(Optional) Rover platform for motor control

**How to Run**

Connect your camera

Place AprilTags in front of the camera

Run the script:

python April_Detect.py


Press q to exit

--------------------------------------------------------------------------------------------------------------------------------------------------------------------


