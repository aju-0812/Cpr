ROVER POCS:



FOCUSED MODULES:

Camera view and April tag detection.

Local planning & global planning- SLAM mapping.

Obstacles avoidance (i.e., rocks and craters etc..).

Finding excavation zone.

Sand depth identification and positioning bucket.

Bucket lift up and sand holding.

Finding construction zone.

Bucket lift down and sand dropping.

Looping lifting and dropping process.

Reaching the start point finally.



CAMERA FUNCTIONS:



April tag detection and identification the process of rover based on the detected tag.

3D output from stereo vision of the camera for SLAM Mapping.

Detection of rocks, sand, craters, walls, other construction zone obstacles.

Depth identification from the obstacles. also, depth of the sand for sand lifting process.



ALGORITHMS \& CODE REQUIRED:



For global path planning Hybrid A\* or A\* algorithm.

For local path planning SLAM mapping.

For navigation DWA - Dynamic Window Approach.

Object detection using TENSOR Flow lite.

Depth identification for both obstacles and sand.

April Tags detection and identification.

Bucket functioning methodology(Lifting and Dropping).





JETSON ORIN NANO FUNCTIONS:



Communication with OAK-D Lite Stereo Depth Camera.

Communication with STM 32 Micro-controller,

&nbsp;       1.Camera stepper motor controller (signals to STM 32 as analog or percent of speed for revolution or degree of turn).

&nbsp;       2.Bucket  stepper motor controller(signals to STM 32 as analog or percent of speed for revolution or degree of turn).

&nbsp;       3.Bucket linear actuator controller(lifting and dropping process of the bucket).

&nbsp;       4.Wheel motor control(forward and backward movement speed , turn left and right speed and rotation of left and right side wheel based on turn(i.e., for right turn - left side motors run parallel forward and right side motors run parallel backward. speed distribution is mandatory for all the movements )

&nbsp;       5.wheel control is based on the navigation command from the navigation algorithm



HARDWARE STACK:

jetson orin nano, OAK-D Lite



COMMUNICATION HARDWARES:

STM-32,Motors(encoders for all motors)



SOFTWARE STACK:

Ubuntu 22.04 Linux system







