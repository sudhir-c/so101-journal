"""
Train a SAC policy on the SO-101 reach env (real hardware, MPS).

Checkpoints the model AND the replay buffer every N steps so a USB disconnect is
RESUMABLE, not a restart. Re-running resumes from the latest checkpoint by
default (`--fresh` to start over). Ctrl-C saves an interrupt checkpoint and
releases the arm (torque off).

    .venv/bin/python -m teleop.rl_reach.train_reach --timesteps 20000
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from teleop.rl_reach.so101_reach_env import SO101ReachEnv

DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / "checkpoints"


def _latest_checkpoint(save_dir: Path, prefix: str):
    """Return (model_zip, replay_buffer_pkl) for the newest checkpoint, or (None, None)."""
    zips = glob.glob(str(save_dir / f"{prefix}_*_steps.zip"))

    def steps(p: str) -> int:
        m = re.search(r"_(\d+)_steps\.zip$", p)
        return int(m.group(1)) if m else -1

    zips = [z for z in zips if steps(z) >= 0]
    if not zips:
        return None, None
    latest = max(zips, key=steps)
    rb = save_dir / f"{prefix}_replay_buffer_{steps(latest)}_steps.pkl"
    return latest, (str(rb) if rb.exists() else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    ap.add_argument("--save-freq", type=int, default=1000, help="checkpoint every N steps")
    ap.add_argument("--prefix", default="reach_sac")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    args = ap.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(SO101ReachEnv())
    ckpt_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(args.save_dir),
        name_prefix=args.prefix,
        save_replay_buffer=True,
    )

    try:
        ckpt, rb = (None, None) if args.fresh else _latest_checkpoint(args.save_dir, args.prefix)
        if ckpt:
            print(f"Resuming from {ckpt}")
            model = SAC.load(ckpt, env=env, device=args.device)
            if rb:
                model.load_replay_buffer(rb)
                print(f"Loaded replay buffer {rb}")
            reset_num_timesteps = False
        else:
            print("Starting fresh")
            model = SAC(
                "MlpPolicy", env, device=args.device, verbose=1,
                learning_starts=500, buffer_size=100_000, batch_size=256,
                train_freq=1, gradient_steps=1,
            )
            reset_num_timesteps = True

        try:
            model.learn(
                total_timesteps=args.timesteps,
                callback=ckpt_cb,
                reset_num_timesteps=reset_num_timesteps,
            )
            model.save(str(args.save_dir / f"{args.prefix}_final"))
            print(f"Saved final model to {args.save_dir / f'{args.prefix}_final'}.zip")
        except KeyboardInterrupt:
            print("\nInterrupted — saving interrupt checkpoint")
            model.save(str(args.save_dir / f"{args.prefix}_interrupt"))
            model.save_replay_buffer(str(args.save_dir / f"{args.prefix}_interrupt_rb"))
    finally:
        env.close()  # disconnect → torque off


if __name__ == "__main__":
    main()
