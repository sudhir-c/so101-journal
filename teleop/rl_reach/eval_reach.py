"""
Roll out a trained reach policy and watch it reach (real hardware).

    .venv/bin/python -m teleop.rl_reach.eval_reach --model teleop/rl_reach/checkpoints/reach_sac_final
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC

from teleop.rl_reach.so101_reach_env import SO101ReachEnv

DEFAULT_MODEL = Path(__file__).resolve().parent / "checkpoints" / "reach_sac_final"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    env = SO101ReachEnv()
    model = SAC.load(str(args.model), device=args.device)
    try:
        successes = 0
        for ep in range(args.episodes):
            obs, info = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            successes += int(info["is_success"])
            print(f"ep {ep:2d}: final dist {info['distance'] * 1000:5.0f} mm  success={info['is_success']}")
        print(f"\nsuccess {successes}/{args.episodes}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
