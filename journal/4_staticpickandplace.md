# 4 - Pick and Place

With the learnings from the previous entry, I trained a policy that could successfully pick and place. I kept the target box location constant this time, bumped up to 40 demonstrations, and trained for 100k steps.

After seeing how long 20k steps took on MPS, I switched to Google Colab Pro and ran on an A100 GPU — though it still took essentially the whole night. I kept my laptop open all night so my Colab session wouldn't disconnect 😂.

I trained another ACT policy. On evaluation with both the object and box in their fixed positions, it succeeded 5/5 trials and the motion was noticeably smoother than my first attempt.

https://github.com/user-attachments/assets/78eedbe5-c471-46a8-bbc1-52417aa589da

https://github.com/user-attachments/assets/7b6188a0-24b7-40a4-9e43-2f4593aefa8c

When I varied the object position, the arm didn't move at all — the input is out of distribution, so the policy has no idea what to do. In another trial, the arm went to the original object location, failed to grasp, and then attempted the grasp again rather than proceeding to the drop location. That last detail is interesting: if the policy were just continuing to execute its learned trajectory, it would have moved toward the box. The fact that it looped back to retry the grasp suggests it developed some implicit notion of whether or not it had the object.

https://github.com/user-attachments/assets/ececbf39-af8e-439a-9005-f6100b818b90
