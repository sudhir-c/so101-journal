# 4 - Pick and place

With the learnings from the previous entry, I sought to train another policy that could successfully pick and place. This time, I ensured my target box location was constant.
I did 40 trials and trained with 100k steps. I trained another ACT policy. 



https://github.com/user-attachments/assets/78eedbe5-c471-46a8-bbc1-52417aa589da

https://github.com/user-attachments/assets/7b6188a0-24b7-40a4-9e43-2f4593aefa8c


The policy works great when I keep the object and box placement constant. It's pretty smooth. 
When I tried to vary the position of the object to be gripped, the arm did not move at all. Since this is out of distribution, the policy does not know what to do.
In another case, the robot still tried to pick up the object at the original location, and failed. However, after failing, it tried to pick the object up again, which shows that the policy developed some notion of having versus not having the object.

https://github.com/user-attachments/assets/ececbf39-af8e-439a-9005-f6100b818b90
