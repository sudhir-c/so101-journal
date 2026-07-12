# 05 - Pick and place with randomized object placement

Having successfully developed a policy that could reliably pick and place in two constant locations, I wanted to develop a more general policy that could adapt to the object being in a randomized location. I collected 40 demonstrations across an arbitrary ~10in x 8in workspace and trained for 100k steps on Colab's A100 — same setup as before, 9 hours of training. The results were suboptimal 💔.

https://github.com/user-attachments/assets/9021ca5d-e954-4ae4-91cc-ddb6a88454e5

The policy failed on every evaluation trial. The arm consistently moved above the workspace but failed to descend and grasp the object, instead stalling and jittering above the table. Since the object position varied between demonstrations but the policy's behavior remained largely the same regardless of object position, my interpretation is that the policy failed to learn a robust mapping from object position to the appropriate grasp trajectory.

I think 40 demonstrations spread across a 10in x 8in space was insufficient coverage, but I think the bigger issue was in how I collected my data. Here is a sped-up video of some of my demonstrations:

https://github.com/user-attachments/assets/6096401d-95c4-4290-ace5-17c10847b098

I move the arm directly to the object and then immediately move from the object to the dropoff box in one continuous motion. I'm wondering whether this makes the task harder for the policy to learn, since there are relatively few frames corresponding to the critical stages of the task (approaching, grasping, and lifting the object). Additionally, the lack of a consistent sequence of intermediate states across demonstrations may make learning harder — visually, each demonstration appears as a single smooth trajectory rather than a sequence of distinct phases. It may be beneficial to collect demonstrations with a more standardized approach strategy — for example, consistently moving above the object before descending to grasp — so that the policy sees a more consistent perception-to-action mapping across demonstrations.

This aligns with ideas presented in [this paper](https://proceedings.mlr.press/v205/gandhi23a.html), which discusses the importance of consistency in demonstrations with respect to the strategy used to complete a task. Seeing the policy fail and replaying my demonstrations helped me understand the idea of a "task strategy" on a deeper level — humans take the nuances of basic manipulation tasks for granted, and I want to make sure I'm not doing the same.

**Postscript:** As my videos got longer and more numerous, I kept hitting GitHub's 10MB attachment limit for markdown files. I built a small [desktop tool](../scripts/video_compressor.py) — a tkinter app wrapping ffmpeg — that compresses clips to fit under the limit, with optional 2x/5x speedup and an auto-generated speed badge overlay.
