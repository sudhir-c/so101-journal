# 8 - Fourth attempt at pick and place with randomized object placement

As described in the previous entry at this point I had discovered that there was a lot of wiggle room in the elbow joint. I did not document how much but it must have been +/- 5 degrees. After tightening this joint i decided to collect another set of data for a new policy.

I only ended up doing 40 trials, as during the midpoint of the trials I realized that I was being inconsistent and jerky with the manner I was moving the arm in. Tightening the joint gave the robot a much different feel, which changed my teleoperation. Additionally, I switched my "target" object. Both objects were rolled up and taped pieces of paper, but the new one I switched to seemed to have less friction and be more resistant to folding, making it more difficult to grip. As I started collecting data, I kept failing on trials and I predicted that whatever policy I ended up with wouldn't be great. My hypothesis was confirmed below:

https://github.com/user-attachments/assets/81b1e29a-47e2-4180-9f09-45fbebf2db53

For fun I figured I would also try fine-tuning SmolVLA using my demonstrations. Unlike ACT, which I trained from scratch on only my ~40 episodes, SmolVLA is a Vision-Language-Action model that comes pretrained on large amounts of vision, language, and robot data — so I'm fine-tuning an existing model rather than teaching one from nothing. It's also language-conditioned, meaning it takes a text instruction as input. I fine-tuned it on the same dataset with the prompt: "Put the blue paper in the bin."

https://github.com/user-attachments/assets/fc3664d9-6bcb-4dff-81e9-03b53483a64b

The resulting policy was very jerky and performed poorly. I looked into why, and there were a few compounding reasons:
- **Inference couldn't keep up with the control loop.** During eval I got warnings about not keeping up with 30fps. SmolVLA is a ~450M-parameter model, roughly 10x larger than ACT, and running it in real time on my MacBook was too slow. When inference lags, the arm executes stale actions, which shows up as stuttering motion regardless of how good the policy actually is. A VLA's size isn't free — it costs you at inference time, not just during training.
- **SmolVLA expects multiple cameras, and I only had one.** The pretrained model was built around a three-camera setup (I had to rename my single camera to fit one of its expected slots). Multiple viewpoints are a big part of how a model like this perceives depth and disambiguates where things are in space. Running it on a single view means the pretrained representations it relies on are working with much less information than they were designed for, which likely hurt its spatial precision on the grasp — and fine-tuning on only 40 episodes isn't enough to make up that gap.
- **The data was thin and inconsistent.** 40 jerky demonstrations is very little to fine-tune a model this large. ACT, being ~10x smaller, tolerates a small, imperfect dataset far better, which is probably why my ACT policies never looked this rough even on flawed data.

Overall I think I really got an appreciation for how finicky demonstration data can be through these trial and errors. Obviously whatever hardware and software I am working with is not state of the art but I think it still reflects how difficult developing a generalist model is.
