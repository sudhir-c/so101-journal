# 07 - Third attempt at pick and place with randomized object placement

As described in the [previous entry](https://github.com/sudhir-c/so101-journal/blob/main/journal/06_movingobjectv2.md), I was able to develop a policy which went through the correct macro-steps to pick up the object, and generally succeeded. I did not collect specifics on the success rate, since I was still noticing some failures so I wanted to improve the policy before I took specific numbers.

I noticed the failures of the previous policy occurred mainly during the grasping step. The robot would pause at the general waypoint above the pickup zone, and as it lowered toward the object, the gripper would be slighlty misaligned, leading to an unsuccessful grasping motion. My hypothesis was that re-training with more trials would fix this, as the additional data would better "teach" the policy how to complete a successful grasping motion. 

Thus, I added 80 more trials to my previous dataset and retrained with 100k steps. Here were the results.

https://github.com/user-attachments/assets/a9baede4-84fb-4761-afb2-4590f688133c

I got lucky with some of the failures, as it slid the paper over while it was still in the upright position, leading the retry to be successful. Counting those as successes, I had a success rate of 90% (70% if you do not count the knock overs to the total tally).

https://github.com/user-attachments/assets/de57267f-554e-4d3d-b123-0e4210dbff56

On this run, I ended with a success rate of 50%. Suboptimal 💔.

At this point I noticed that the wrist joint on my follower had some slack and could wiggle about +/- 2 degrees. Unfortunately, I forgot to get a video of the movement before a tightened the joint. With that being done, I tested the policy again. 

https://github.com/user-attachments/assets/30bfe939-4c78-4c93-b2f9-4ce852ee90f1

Counting some of the knock-overs as succcesses, this was still only a 60% success rate. 

I am pretty sure that the wiggle room in the joint had been present since assembly, so I decided not to trust the data I had collected and by extension the models I had trained. It's possible the mismatch between the positions recorded and the actual physical position of the arm had enough of a discrepancy created noise within the training data, degrading the policy's precision. I think it would be unwise to assume that this is not a source of error for the current policy, and try to make this policy more accurate my adding more demonstrations, or augmenting it in some other way.

In general, I think restarting data collection is not the worst outcome. Having now performed 160 pick-and-place trials, I think my data quality as a demonstrator will be better and more smoother, which will hopefully lead to a more robust policy. Moreover, I think I will be able to do a more thorough job ensuring that I have data coverage of the full pickup space.






