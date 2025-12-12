#!/usr/bin/env python3
import depthai as dai
import numpy as np
import cv2
from ultralytics import YOLO

# ------------------ Load YOLOv8 Model ------------------
print("Loading YOLOv8 model...")
model = YOLO("yolov8s.pt")

# ------------------ Colorize Depth Function ------------------
def colorize_depth(depth_frame):
    depth8 = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
    depth8 = depth8.astype(np.uint8)
    return cv2.applyColorMap(depth8, cv2.COLORMAP_JET)

# ------------------ Create Pipeline ------------------
pipeline = dai.Pipeline()

# RGB camera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(640, 360)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

# Stereo depth
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)

# ------------------ Run Device ------------------
with dai.Device(pipeline) as device:

    qRgb = device.getOutputQueue("rgb", 4, False)
    qDepth = device.getOutputQueue("depth", 4, False)

    print("YOLOv8 + Depth Map + Bounding Box Running... Press Q to exit.")

    while True:
        rgb_frame = qRgb.get().getCvFrame()
        depth_frame = qDepth.get().getFrame()

        results = model(rgb_frame, verbose=False)
        detections = results[0]

        depth_color = colorize_depth(depth_frame)

        for box in detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Depth (millimeters)
            if 0 <= cx < depth_frame.shape[1] and 0 <= cy < depth_frame.shape[0]:
                distance_mm = depth_frame[cy, cx]
            else:
                distance_mm = 0

            if distance_mm > 0:
                distance_m = distance_mm / 1000.0
                text = f"{label} {distance_m:.2f} m"
            else:
                text = f"{label} ---"

            # -------- DRAW ON RGB WINDOW --------
            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(rgb_frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # -------- DRAW ON DEPTH MAP --------
            cv2.rectangle(depth_color, (x1, y1), (x2, y2), (0,255,0), 2)

            # Center dot
            cv2.circle(depth_color, (cx, cy), 5, (0,0,255), -1)

            # Distance text on depth window
            if distance_mm > 0:
                cv2.putText(depth_color, f"{distance_m:.2f} m",
                            (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,255,0), 2)

        # Display
        cv2.imshow("YOLOv8 RGB + Distance", rgb_frame)
        cv2.imshow("Depth Map", depth_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
