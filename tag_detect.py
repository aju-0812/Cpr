import cv2
import numpy as np
import depthai as dai
from pupil_apriltags import Detector

# ------------------ Create Pipeline ------------------
pipeline = dai.Pipeline()

# Mono Cameras (OV7251 supports only 400p or 480p)
monoLeft = pipeline.create(dai.node.MonoCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutLeft.setStreamName("left")

monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoLeft.out.link(xoutLeft.input)

# RGB Camera
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")

camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setPreviewSize(640, 360)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.preview.link(xoutRgb.input)

# ------------------ Draw AprilTags ------------------
def draw_tags(image, tags):
    for tag in tags:
        center = (int(tag.center[0]), int(tag.center[1]))
        corners = tag.corners.astype(int)

        cv2.circle(image, center, 5, (255, 0, 255), 2)

        for i in range(4):
            cv2.line(image,
                     tuple(corners[i]),
                     tuple(corners[(i + 1) % 4]),
                     (0, 255, 0), 2)

        cv2.putText(image, f"ID:{tag.tag_id}",
                    (center[0] - 30, center[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)
    return image

# AprilTag detector
at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    decode_sharpening=0.25,
)

# ------------------ Run Pipeline ------------------
with dai.Device(pipeline) as device:
    qLeft = device.getOutputQueue("left", 4, False)
    qRgb = device.getOutputQueue("rgb", 4, False)

    while True:
        leftFrame = qLeft.get()
        rgbFrame = qRgb.get()

        left_img = leftFrame.getCvFrame()
        rgb_img = rgbFrame.getCvFrame()

        # Convert grayscale mono to 3-channel BGR
        left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)

        # Detect AprilTags (use first channel of mono)
        tags = at_detector.detect(left_img[:, :, 0], estimate_tag_pose=False)
        left_img = draw_tags(left_img, tags)

        # Resize both images to same size
        left_resized = cv2.resize(left_img, (640, 360))
        rgb_resized = cv2.resize(rgb_img, (640, 360))

        # Combine images horizontally
        combined = np.hstack((left_resized, rgb_resized))

        cv2.imshow("OAK-D Lite AprilTag | Left + RGB", combined)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
