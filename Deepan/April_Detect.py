
import cv2
import numpy as np
from pupil_apriltags import Detector

# -------- SEQUENTIAL NAVIGATION ----------
NAV_SEQUENCE = [2, 3, 1]
current_stage = 0
mission_complete = False

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


# -------- APRILTAG DETECTOR ----------
at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    decode_sharpening=0.25
)


# -------- DRAW TAG ----------
def draw_tag(image, tag, label, direction):
    center = (int(tag.center[0]), int(tag.center[1]))
    corners = tag.corners.astype(int)

    for i in range(4):
        cv2.line(image, tuple(corners[i]),
                 tuple(corners[(i + 1) % 4]),
                 (0, 255, 0), 2)

    cv2.circle(image, center, 6, (0, 0, 255), -1)

    cv2.putText(image, label,
                (center[0] - 50, center[1] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.putText(image, direction,
                (center[0] - 50, center[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return image


# -------- CAMERA INPUT ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting AprilTag Sequential Navigation...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # AprilTag detection
    tags = at_detector.detect(gray)
    target_id = NAV_SEQUENCE[current_stage]
    status_msg = f"LOOKING FOR TAG {target_id}"
    detected_tag = None

    for tag in tags:
        if tag.tag_id == target_id:
            detected_tag = tag
            break

    if detected_tag is not None:
        cx = int(detected_tag.center[0])
        direction = get_direction(cx)

        print(f"[FOUND] Tag {target_id}, CX={cx}, Direction={direction}")

        label = f"ID {target_id} DETECTED"
        frame = draw_tag(frame, detected_tag, label, direction)

        current_stage += 1

        if current_stage >= len(NAV_SEQUENCE):
            mission_complete = True
            current_stage = len(NAV_SEQUENCE) - 1
            print("\n🎉 Mission Complete – All zones reached!\n")

    # Display status on single page
    cv2.putText(frame, status_msg, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if mission_complete:
        cv2.putText(frame, "MISSION COMPLETE!", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # SINGLE WINDOW DISPLAY
    cv2.imshow("AprilTag Navigation", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
