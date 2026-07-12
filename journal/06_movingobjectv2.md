# 06 - Second attempt at pick and place with randomized object placement

As described in the [previous entry](https://github.com/sudhir-c/so101-journal/blob/main/journal/05_movingobject.md), my first attempt at developing a policy that could adjust to a variable object pickup point was unsuccessful, and I suspected that it was because of the continuous way I moved the arm. 
For this policy, I defined a clear pickup region with tape, so I could be visually sure that I was collecting data uniformly across my whole pickup region. 
Most importantly, in each demonstration, I made the effort of explicitly splitting my demonstration up into 6 concrete steps. My hope was that being intentional with taking this steps would avoid the sort of continous motion that might make sense to me as a human but be hard to infer a policy from. 
1) Move the arm to a central spot above the pickup region. 
2) Open the gripper as I lower the arm and grasp the object. 
3) Move the object back up above the pickup region. 
4) Sweep the arm over so it is above the dropoff box. 
5) Open the grippers, dropping the object. 
6) Move the arm back to its home state. 

For good measure, I also doubled the number of trials from 40 to 80. I trained on 100k steps on the same Colab setup as earlier.

The results were much better! The policy was still not perfect, but the general actions being taken were correct. I saw the robot generally taking the same 6 steps I defined in data collection, and whatever failure that occured was merely the robot being imprecise in the grasping motion and missing the object. It was also good to see that when the robot failed to pick up the object, the policy would recognize that it did not have the object and retry the pickup. 

https://github.com/user-attachments/assets/109f79e7-23b4-412d-88fb-bedacc8d8f51

I was wondering if the imprecision in the pickup motion was more common in the farther pickups, where I as the demonstrator probably went slower and was not as smooth in controlling the arm. 

https://github.com/user-attachments/assets/914b17d5-35db-4078-a245-84acfaadb488

https://github.com/user-attachments/assets/99391f35-b62f-4aa2-a08a-b20219101154

It seems like in both cases, there are pickup failures. Because the policy is taking the correct macro-action, and the failure is more in the details, my hypothesis is that I can increase the policy success rate with more demonstrations. 


