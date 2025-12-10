import depthai as dai
import cv2

# Create pipeline
pipeline = dai.Pipeline()

# Color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Depth camera
cam_mono_left = pipeline.createMonoCamera()
cam_mono_right = pipeline.createMonoCamera()

cam_mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
cam_mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo = pipeline.createStereoDepth()
cam_mono_left.out.link(stereo.left)
cam_mono_right.out.link(stereo.right)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# Connect device
with dai.Device(pipeline) as device:
    rgbQueue = device.getOutputQueue("rgb")
    depthQueue = device.getOutputQueue("depth")

    while True:
        rgbFrame = rgbQueue.get().getCvFrame()
        depthFrame = depthQueue.get().getFrame()

        cv2.imshow("RGB", rgbFrame)
        cv2.imshow("Depth", depthFrame)

        if cv2.waitKey(1) == ord('q'):
            break
