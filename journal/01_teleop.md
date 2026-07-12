# 01 - Teleop

With both arms assembled and motors set up, the next milestone was teleoperation — getting Spectre (the follower) to copy whatever I do to Phantom (the leader) in real time.

The way it works: Phantom's motors run with torque disabled, so I can freely backdrive the joints by hand. At 60Hz, LeRobot reads the angular position of each of Phantom's 6 joints over the servo bus and writes those as target positions to Spectre's corresponding motors. The follower then uses each servo's onboard position controller to track those targets.

Before any of that though, I had to calibrate — moving each joint to its physical min and max so LeRobot knows the range of motion and can prevent the arm from commanding positions that would damage itself. After that I ran [`lerobot-teleoperate`](https://github.com/huggingface/lerobot):

https://github.com/user-attachments/assets/b0a450b3-0c23-4426-a6f3-bf168089c963

It wasn't totally smooth, though. Every now and then on startup, not all motors get detected on the servo bus — the arm comes up partially mapped, or the connection fails entirely. Unplugging and replugging the USB-C fixes it. I have enough experience with serial buses from FRC to know this kind of flakiness isn't unusual, so I decided not to rabbit-hole on the root cause and just move on.
