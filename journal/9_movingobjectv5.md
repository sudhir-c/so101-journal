# 9 - Fifth attempt at pick and place with randomized object placement

For my next attempt I made the size of the pickup area smaller, and changed the target object to pickup from a taped up piece of paper to an eraser. Both of these factors definitely made the pickup task easier. A smaller pickup space meant the arm did not have to extend as far, and the eraser was much bigger, grippier, and heavier than the paper, leading to a higher margin of error with the pickup. 
I think these factors were not only making it difficult for the inferred policy to act, but also made it difficult for me as a demonstrator to provide good data. I could definitely be better at teleoperating the arm and do a smoother job with all the pickups, but because of the previously mentioned factors I think my data had some jerkier motion. It could just be confirmation bias but I get the sense that I will have a better policy if I provide demonstrations where each "macro-action" occurs smoothly and in one contiguous motion. 

The policy performed quite well, with a 19/20 success rate overall. After 10 trials I realized the whole end effector mechanism was loose, so I tightened it.

https://github.com/user-attachments/assets/b8b26ebd-ac09-41d4-962d-d0d69dca9f64

https://github.com/user-attachments/assets/0b29c9a4-0584-4896-b941-71999e14bd9b

Out of curiosity I wanted to see how this policy would respond to out-of-distribution states. I varied the orientation of the eraser and placed it outside the tape box. I was thinking that it might be able to generalize if the eraser was placed outside the tape box, as it would be the same macro steps to complete the task but only a different pickup pose. I did not expect the policy to successfully react to eraser orientation changes, as it would require a fundamentally different motion for the arm to pick it up. However, I was thinking that just rotating it upright might work as the policy probably might consider that state close enough to the one where the eraser is oriented normally. 

https://github.com/user-attachments/assets/6295295f-f6ee-4818-a5fc-355d217a1ddb

My prediction on orientation was correct, but putting the eraser outside/on the tape line caused it to fail. Because the tape was a prominent visual feature it's possible that the policy learned not to consider anything in those pixels as a viable pickup pose.
