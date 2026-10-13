# simple-apriltag-detector
Supports 36h11 and 41h12 tags 

```python
## First verison of the apriltag dector for the IEEE robotics contest
# By: Joshua Adcock
# Last updated on: 10/12/26 

## Possible workflow
# camera setup > tag detection > 6 dof state estimation > output (x, y, z) and (roll, pitch, yaw) over serial

import cv2

## <<<<< For 36h11 tags >>>>> ## 
from pupil_apriltags import Detector
detector = Detector(families="tag36h11")

## <<<<< FOr 42h12 tags >>>>> ## 
#from pyapriltags import Detector
#detector = Detector(families="tagStandard41h12")

from scipy.spatial.transform import Rotation

camera = cv2.VideoCapture(0)

# Replace values with actual camera metrics
fx, fy = 600.0, 600.0
cx, cy = 320.0, 240.0

TAG_SIZE = 0.080  # in meters; measure the tag's black/white boundary

while True:
    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tags = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=TAG_SIZE
    )

    for tag in tags:
        position = tag.pose_t.flatten()
        rotation = Rotation.from_matrix(tag.pose_R)
        rpy = rotation.as_euler("xyz", degrees=True)

        x, y = tag.center.astype(int)

        # draw box around apriltag
        cv2.polylines(
            frame, [tag.corners.astype(int)], True, (0, 255, 0), 2
        )

        # print tag ID
        cv2.putText(
        frame,
            f"ID {tag.tag_id}",
            (x, y - 0),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 255, 0), 2
        )

        # print (x, y, z) location
        cv2.putText(
            frame,
            f"XYZ: {position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f} m",
            (x, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1
        )

        # print roll, pitch, yaw rotation
        cv2.putText(
            frame,
            f"Rotation: {rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f} m",
            (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1
        )

        print(
            f"Tag {tag.tag_id}: "
            f"position={position.round(3)} m, "
            f"roll/pitch/yaw={rpy.round(1)} degrees"
        )

    cv2.imshow("AprilTag State Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
```

You will need to install the following libraries: 
```python
pip install opencv-python pupil-apriltags pyapriltags
```
