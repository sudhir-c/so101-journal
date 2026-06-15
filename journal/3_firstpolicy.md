# 3 - First policy

I scrunched up a small teabag and tore the lid off one of the motor boxes to get my setup for making my first policy. 
I figured I would start with a basic pick and place task, and I only stuck to 10 trials as I just wanted to verify that I could sucessfully train a policy.
I trained an ACT policy with 20k steps.

https://github.com/user-attachments/assets/c4009bb8-28c6-4403-86e9-27d74351b068

Although it picked up the object, the arm motion is very jerky and it missed the box. 
I think the former is due to the fact that I only did 5 trials. 
As for the latter, I think I may have nudged the box between training and evaluation which would obviously cause problems, given that my policy is not (yet) robust enough to handle objects moving around. 
I also noticed after training was done that I accidentally moved the leader arm into the camera frame during training. The end effector and object were still visible, but I changed my setup to eliminate that potential source of error in my data.









