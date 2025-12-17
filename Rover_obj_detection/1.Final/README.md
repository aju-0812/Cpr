# 🚜 Caterpillar-AI Autonomous Rover: Vision System (Final Package)

**Version:** 1.0 (Final Release)  
**Date:** December 2025  
**Developer:** Caterpillar-AI Team

---

## 📖 Overview
This package contains the complete, deployable computer vision system for the Caterpillar-AI Autonomous Rover. It includes trained **YOLOv8 models** for obstacle detection and **Streamlit applications** for both high-performance PCs and resource-constrained hardware (Raspberry Pi/Jetson).

The system performs:
1.  **Object Detection**: Identifies obstacles (Rocks, Walls, etc.) in real-time.
2.  **Virtual Depth Mapping**: Converts 2D camera images into 1D depth arrays for navigation.
3.  **Path Planning**: Calculates the safest "gap" and steering angle.

---

## 📂 Directory Structure & Important Files

```text
1.Final/
├── app_main.py              # 🖥️ PC Application (Full Features, PyTorch)
├── app_lite.py              # ⚡ Hardware Application (Fast, TFLite/ONNX)
├── requirements.txt         # 📦 Python Dependencies
├── models/                  # 🧠 The "Brain" Files
│   ├── yolo_fast.pt         # [PC] Fast PyTorch Model (6.2 MB)
│   ├── yolo_accurate.pt     # [PC] High-Accuracy PyTorch Model (19 MB)
│   └── yolo_quantized.tflite# [HW] Optimized Edge Model (3.2 MB) - USE THIS FOR ROVER!
└── training_scripts/        # 🎓 Scripts used to train the models
    ├── train_yolo.py
    └── train_fast.py
```

---

## 🧠 Model Details

| Model Filename | Format | Size | Speed | Accuracy | Best Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`yolo_fast.pt`** | PyTorch (`.pt`) | 6.2 MB | Fast | Medium | Testing on Laptop/PC |
| **`yolo_accurate.pt`** | PyTorch (`.pt`) | 19 MB | Slow | High | High-end PC Demos |
| **`yolo_quantized.tflite`** | TFLite (`.tflite`) | **3.2 MB** | **Ultra Fast** | Medium | **Raspberry Pi / Jetson (DEPLOYMENT)** |

> **Note:** The `yolo_quantized.tflite` model uses INT8 quantization, making it ~4x faster on ARM CPUs (like Raspberry Pi) compared to standard models.

---

## 🚀 How to Run

### 1. Installation
First, install the required libraries.
```bash
pip install -r requirements.txt
```

### 2. Running on PC / Laptop (Testing)
If you are testing on your Windows/Linux laptop, use the main app. It defaults to the `yolo_fast.pt` model.
```bash
streamlit run app_main.py
```

### 3. Running on Rover Hardware (Deployment)
**This is the mode for the actual robot.** It uses the lightweight `yolo_quantized.tflite` model.
```bash
streamlit run app_lite.py
```

---

## ⚙️ How It Works (The Logic)

1.  **Input**: The app takes a video feed or image.
2.  **Detection**: The YOLO model finds bounding boxes for obstacles.
3.  **Depth Mapping**:
    *   The image is divided into **32 vertical sectors**.
    *   The "depth" of each sector is estimated based on the Y-coordinate of the lowest obstacle in that column (lower in image = closer).
4.  **Smoothing**: A temporal filter smooths out jittery readings.
5.  **Decision**: The rover looks for the widest, deepest gap in the sectors and steers towards it.

## 🛠️ Troubleshooting

*   **"Model not found"**: Ensure the `models/` folder exists and contains the `.pt` or `.tflite` files.
*   **Slow Performance**: Switch to `app_lite.py` or reduce the "Navigation Sectors" slider in the sidebar.
*   **Camera Issues**: If using a webcam, ensure no other app is using it. You may need to change `cv2.VideoCapture(0)` index in the code.
