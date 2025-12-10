#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
from pupil_apriltags import Detector

# ----------- NAVIGATION SEQUENCE ------------
NAV_SEQUENCE = [2, 3, 1]
current_stage = 0
mission_complete = False

# ----------- DIRECTION THRESHOLDS ----------
FRAME_WIDTH = 640
CENTER_X = FRAME_WIDTH // 2
LEFT_THRESHOLD = CENTER_X - 40
RIGHT_THRESHOLD = CENTER_X + 40


def get_direction(tag_x):
    """Given tag center X, return direction for rover."""
    if tag_x < LEFT_THRESHOLD:
        return "TURN LEFT"
    elif tag_x > RIGHT_THRESHOLD:
        return "TURN RIGHT"
    else:
        return "GO STRAIGHT"


# ----------- DEPTHAI PIPELINE ------------
pipeline = dai.Pipeline()

# Left Mono Camera
monoLeft = pipeline.create(dai.node.MonoCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutLeft.setStreamName("left")

monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoLeft.out.link(xoutLeft.input)

# RGB Camera (display only)
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")

camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setPreviewSize(640, 360)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.preview.link(xoutRgb.input)

# ----------- APRILTAG DETECTOR ------------
at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    decode_sharpening=0.25
)


# ----------- DRAW TAG DETAILS ------------
def draw_tag(image, tag, label, direction):
    center = (int(tag.center[0]), int(tag.center[1]))
    corners = tag.corners.astype(int)

    # Box outline
    for i in range(4):
        cv2.line(image,
                 tuple(corners[i]),
                 tuple(corners[(i + 1) % 4]),
                 (0, 255, 0), 2)

    # Center point
    cv2.circle(image, center, 6, (0, 0, 255), -1)

    # ID label
    cv2.putText(image, f"{label}",
                (center[0] - 50, center[1] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Direction label
    cv2.putText(image, f"{direction}",
                (center[0] - 50, center[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return image


# ----------- RUN PIPELINE ------------
with dai.Device(pipeline) as device:

    qLeft = device.getOutputQueue("left", 4, False)
    qRgb = device.getOutputQueue("rgb", 4, False)

    while True:
        leftFrame = qLeft.get()
        rgbFrame = qRgb.get()

        left_img = cv2.cvtColor(leftFrame.getCvFrame(), cv2.COLOR_GRAY2BGR)
        rgb_img = rgbFrame.getCvFrame()

        # Detect AprilTags
        tags = at_detector.detect(left_img[:, :, 0])

        # Current target ID
        target_id = NAV_SEQUENCE[current_stage]

        status_msg = f"LOOKING FOR TAG {target_id}"
        detected_tag = None

        # Only detect tag of current stage
        for tag in tags:
            if tag.tag_id == target_id:
                detected_tag = tag
                break

        # If correct tag detected
        if detected_tag is not None:

            cx = int(detected_tag.center[0])  # center x-position
            direction = get_direction(cx)

            print(f"[FOUND] Tag {target_id}, Center X = {cx}, Direction = {direction}")

            label = f"ID {target_id} DETECTED"
            draw_tag(left_img, detected_tag, label, direction)

            # Move to next stage after reaching zone
            current_stage += 1

            if current_stage >= len(NAV_SEQUENCE):
                mission_complete = True
                current_stage = len(NAV_SEQUENCE) - 1
                print("\n🎉 Mission Complete – All zones reached!\n")

        # Display status
        cv2.putText(left_img, status_msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if mission_complete:
            cv2.putText(left_img, "MISSION COMPLETE!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Combine views
        left_resized = cv2.resize(left_img, (640, 360))
        rgb_resized = cv2.resize(rgb_img, (640, 360))
        combined = np.hstack((left_resized, rgb_resized))

        cv2.imshow("Sequential AprilTag + Direction", combined)

        if cv2.waitKey(1) == ord('q'):
            break