#!/bin/bash
# Activate your venv (adjust path if different)
source /Users/sudhirc/Desktop/Projects/so-101-arm/.venv/bin/activate

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B3D0486331 \
    --robot.id=spectre \
    --robot.cameras="{ limecam: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5B3D0466471 \
    --teleop.id=phantom \
    --display_data=true