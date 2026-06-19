# 2 - Camera Setup

Having confirmed the arms work, I had to set up my camera to get vision input for the policy.

The camera that ships with the Partabot kit is a standard USB webcam. I wrote a small [`cam_viz.py`](../scripts/cam_viz.py) script using OpenCV to verify the feed and check what resolution the camera actually negotiated — requesting 1920x1080 and printing back whatever the driver reported.

The trickier decision was placement. I wanted the camera elevated and angled rather than flat-on, for two reasons: an angled view gives the image a sense of depth that a top-down shot would lose, and it keeps the full working volume of the arm in frame — including the end effector — without the arm occluding the object during a pick. The setup I landed on gives a clear view of both the gripper and the target area throughout the full trajectory.

<img width="1023" height="948" alt="Screenshot 2026-06-15 at 11 51 12 AM" src="https://github.com/user-attachments/assets/454655f1-8b51-401c-b55f-94f1c0a224d2" />
<img width="1023" height="664" alt="Screenshot 2026-06-15 at 11 52 32 AM" src="https://github.com/user-attachments/assets/ddeb4ad9-e4f8-49dc-b3cf-489a5e384dac" />

During teleoperation, LeRobot streams the camera feed live through [Rerun](https://rerun.io/), which made it easy to confirm the view looked right before collecting any data.
