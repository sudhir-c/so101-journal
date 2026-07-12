# 03 - First Policy

I scrunched up a small teabag and tore the lid off one of the motor boxes to get my setup for a basic pick and place task. I just wanted to verify I could successfully train a policy end to end, so I kept things simple.

I collected 10 demonstrations of ~15 seconds each at 60Hz, then trained an [ACT](https://tonyzhaozh.github.io/aloha/) policy for 20k steps. ACT (Action Chunking with Transformers) predicts a chunk of future actions at once rather than one step at a time, which helps reduce compounding errors during rollout. Training ran on MPS (Apple Silicon) and took overnight.

https://github.com/user-attachments/assets/c4009bb8-28c6-4403-86e9-27d74351b068

The arm picked up the object but the motion was jerky and it missed the box. I think the jerkiness comes down to the small dataset — 10 demos isn't much to learn a smooth trajectory from. The missed drop is likely because I nudged the box between training and evaluation; since the policy has no robustness to object position yet, even a small shift throws it off.

I also noticed after training that I'd accidentally moved the leader arm into the camera frame during some demonstrations. The end effector and object were still visible, but it introduces a spurious visual feature the policy might latch onto. I updated my setup to keep the leader out of frame before collecting any more data.
