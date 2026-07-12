# 00 - Intro
This is the start of me documenting my progress with the [SO-101 robot arm](https://github.com/TheRobotStudio/SO-ARM100). hopefully this works 😛

The SO-101 is a 6-DOF open source robot arm — 5 arm joints plus a gripper — all driven by [Feetech STS3215](https://www.feetechrc.com/STS3215.html) serial bus servos. These communicate over a half-duplex TTL serial bus, meaning all 6 motors share a single wire and take turns talking. I ordered the electronics-only kit from Partabot and had some friends at the University of Washington print the structural parts.

<img width="952" height="685" alt="Screenshot 2026-06-14 at 2 05 35 PM" src="https://github.com/user-attachments/assets/7d330778-a03e-446c-9f08-338d76e3fa3d" />

The assembly went pretty smoothly. The kit comes with two arms — a **leader** and a **follower**. The idea is that you physically move the leader by hand, and the follower mirrors it in real time by reading the leader's joint angles over the servo bus and replaying them on its own motors. This leader-follower setup is the foundation for kinesthetic teaching: you use the leader to demonstrate tasks, and those demonstrations become the training data for an imitation learning policy. I'll be using [LeRobot](https://github.com/huggingface/lerobot) by HuggingFace as my framework throughout this project.

<img width="982" height="723" alt="Screenshot 2026-06-14 at 2 13 47 PM" src="https://github.com/user-attachments/assets/f039ca9e-ee74-4cf5-bec6-16033c160b59" />

I come from a FIRST Robotics Competition background, so I've built arms before — though those were telescoping rather than jointed, and the control was closed-loop (PID) rather than learned. This project is my first time working with learned robot behavior, so I'm excited to see how far I can get.

I named the arms after two of my FRC robots. The leader is **Phantom**, after our 2023 robot:
<img width="920" height="638" alt="Screenshot 2026-06-14 at 2 19 35 PM" src="https://github.com/user-attachments/assets/14eb93fc-e205-48f3-834e-657618e7a16e" />
<img width="1920" height="1280" alt="image" src="https://github.com/user-attachments/assets/16830bb6-14ca-48b1-8cad-e192e9e0ec0c" />

The follower is **Spectre**, after our 2025 robot:
<img width="914" height="643" alt="Screenshot 2026-06-14 at 2 21 30 PM" src="https://github.com/user-attachments/assets/beedd7c6-5605-4b87-b99b-ce1371a1a543" />
<img width="2048" height="1365" alt="image" src="https://github.com/user-attachments/assets/7cf0c315-1719-4602-be8c-ef3a5f0c630d" />
