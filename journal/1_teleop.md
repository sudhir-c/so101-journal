# 1 - Teleop

With both arms assembled and the motors all set up, the next milestone was teleoperation — getting Spectre (the follower) to copy whatever I do to Phantom (the leader) in real time.

I physically move Phantom around with my hand, the leader arm reads the angle of each of its joints over the servo bus, and Spectre mirrors those positions on its own motors. 

After calibrating for min/max motor positions, I ran [`lerobot-teleoperate`](https://github.com/huggingface/lerobot):

https://github.com/user-attachments/assets/b0a450b3-0c23-4426-a6f3-bf168089c963

It wasn't totally smooth, though. The servo bus is sometimes faulty — every now and then when I plug in and start things up, not all of the motors get detected. 
The arm comes up partially mapped (or the connect just fails), and I have to unplug the usb-c connection before the full chain of motors shows up again. 



