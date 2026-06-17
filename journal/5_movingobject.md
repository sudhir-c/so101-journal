# 5 - Pick and place with randomized object placement

Having successfully developed a policy that could reliably pick and place in two constant locations, I wanted to develop a more general policy that could adapt to the object being in a randomized location. 
I did 40 trials across about 10in x 8in space, and trained with 100k steps. The results were suboptimal 💔. 

https://github.com/user-attachments/assets/9021ca5d-e954-4ae4-91cc-ddb6a88454e5

The arm consistently moved above the workspace but failed to descend and grasp the object. Instead, it stalled and jittered above the table. Since the object position varied between demonstrations but the policy's behavior remained largely the same, my interpretation is that the policy failed to learn a robust mapping from object position to the appropriate grasp trajectory.

I think only doing 40 trials for the 10in x 8in space was not enough to develop a robust policy, but I think the larger issue was in how I collected my data.
Here is a sped-up video of some of my runs. 

https://github.com/user-attachments/assets/6096401d-95c4-4290-ace5-17c10847b098

I move the arm directly to the object and then immediately move from the object to the dropoff box in one continuous motion. I'm wondering whether this makes the task more difficult for the policy to learn, since there are relatively few frames corresponding to the critical stages of the task (approaching, grasping, and lifting the object). Additionally, I wonder whether the lack of a consistent sequence of intermediate states across demonstrations makes learning harder. Visually, the demonstrations appear as a single smooth trajectory rather than a sequence of distinct phases. It may be beneficial to collect demonstrations with a more standardized approach strategy—for example, consistently moving above the object before descending to grasp—so that the policy sees a more consistent perception-to-action mapping across demonstrations.

This experience aligns with some of the ideas presented in [this paper](https://proceedings.mlr.press/v205/gandhi23a.html), which discusses the importance of having consistency in the demonstration the robot learns off of, specifically with regards to the strategy used to complete a simple task. Seeing how the policy failed and replaying my demonstrations helped me understand this idea of a "task strategy" on a deeper level. I think this kind of thing is not intuitive at first because humans typically take the nuances behind basic manipulation tasks for granted, but I want to challenge myself not to do this to understand these robotics systems better.
