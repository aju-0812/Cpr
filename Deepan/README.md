# 🚗 AprilTag-Based Sequential Navigation System

## 📌 Overview
This project implements a **vision-based sequential navigation system** using **AprilTags** and **OpenCV**.  
A camera detects AprilTags in real time and guides a rover (or robot) through **predefined navigation zones** by following a fixed tag order.

The system determines whether the rover should **turn left, turn right, or move straight** based on the horizontal position of the detected AprilTag in the camera frame.

---

## ✨ Key Features
- ✅ Real-time AprilTag detection  
- ✅ Sequential navigation using predefined tag IDs  
- ✅ Direction guidance (**LEFT / RIGHT / STRAIGHT**)  
- ✅ Visual overlay of tag boundaries, center point, and commands  
- ✅ Single-window display with mission status  
- ✅ Easily extendable to motor control or ROS integration  

---

## 🔢 Navigation Logic
The rover must detect AprilTags in a **fixed sequence**:

```bash
NAV_SEQUENCE = [2, 3, 1]

---

## 🧰 Requirements
## 🖥️ Software

Install the required Python libraries using pip:

```bash
pip install opencv-python numpy pupil-apriltags

🧱 Hardware

USB Camera / Laptop Camera

Optional: Rover platform for motor control

---

▶️ How to Run

Connect your camera to the system

Place AprilTags in front of the camera

Run the script:

```bash
python April_Detect.py


Press q to exit the application
