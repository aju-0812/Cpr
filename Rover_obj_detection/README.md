# Rover Object Detection & Navigation System 🚜

This module implements an autonomous vision system for a rover using YOLOv8/YOLO11. It detects obstacles (like boulders, craters) and provides real-time navigation commands (Go Straight, Turn Left, Turn Right).

## 📂 Project Structure

- **`app.py`**: A Streamlit web dashboard to upload videos, view detections, and see navigation commands.
- **`train_yolo.py`**: The main training script for high-accuracy models (uses GPU if available, runs for 100 epochs).
- **`train_fast.py`**: A fast training script optimized for CPUs (uses YOLOv8 Nano, 30 epochs, smaller images).
- **`src/detection.py`**: A standalone script to run object detection on your webcam.
- **`src/main.py`**: The core logic combining detection with Reinforcement Learning (RL) navigation.
- **`data/`**: Contains your training and validation datasets (images and labels).
- **`runs/`**: Stores trained models and training logs.

## 🚀 Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Dataset**:
    *   Place your images in `images/` and labels in `labels/` inside this folder.
    *   The training scripts will automatically organize them into `data/train` and `data/validation`.

## 🧠 Training the Model

You have two options for training:

### Option A: High Accuracy (Recommended for GPU)
Trains a robust model with data augmentation for 100 epochs.
```bash
python train_yolo.py
```
*   **Output**: `runs/detect/train/weights/best.pt`

### Option B: Fast Mode (Recommended for CPU)
Trains a lightweight model (Nano) quickly for 30 epochs.
```bash
python train_fast.py
```
*   **Output**: `runs/detect/train_fast/weights/best.pt`

## 🎮 How to Run

### 1. Streamlit Dashboard (Video Testing)
Upload a video to see how the rover would react.
```bash
streamlit run app.py
```

### 2. Live Webcam Detection
Run detection directly on your webcam feed.
```bash
python src/detection.py
```
*   *Note: The scripts automatically look for the best available model (Fast or Normal).*

## 🤖 How it Works

1.  **Vision**: The YOLO model scans the frame for objects defined in `classes.txt`.
2.  **Logic**: The image is split into three zones (Left, Center, Right).
3.  **Decision**:
    *   **Center Clear**: -> "GO STRAIGHT"
    *   **Center Blocked**: Checks Left vs Right zones.
    *   **Left Clear**: -> "TURN LEFT"
    *   **Right Clear**: -> "TURN RIGHT"
    *   **All Blocked**: -> "STOP" or "Caution Turn"
