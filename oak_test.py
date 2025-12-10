from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')         # RGB camera
    stereo = oak.create_stereo()              # creates stereo/depth streams
    stereo.config_stereo(align=color)         # align depth to color

    oak.visualize([color, stereo])            # visualize both streams
    oak.start(blocking=True)
