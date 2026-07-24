"""
Roll out a trained reach policy and watch it reach (real hardware).

By default this also launches a live 3D visualizer (robot camera feed + a
three.js scene of the arm skeleton, target, and end-effector) at
http://127.0.0.1:8091 and opens it in your browser. The visualizer is display
only — the eval process remains the sole owner of the serial port. Disable it
with --no-viz.

    .venv/bin/python -m teleop.rl_reach.eval_reach --model teleop/rl_reach/checkpoints/reach_sac_final
"""

from __future__ import annotations

import argparse
import time
import webbrowser
from pathlib import Path

from stable_baselines3 import SAC

from teleop.rl_reach.fk import fk_chain
from teleop.rl_reach.so101_reach_env import (
    MAX_EE, MIN_EE, SUCCESS_THRESH, SO101ReachEnv,
)

DEFAULT_MODEL = Path(__file__).resolve().parent / "checkpoints" / "reach_sac_final"


def _make_state(info, episode, step, successes, total) -> dict:
    """Plain JSON-serializable snapshot for the visualizer."""
    return {
        "ee": [float(v) for v in info["tip_xyz"]],
        "target": [float(v) for v in info["target_xyz"]],
        "skeleton": fk_chain(info["joints_deg"]),   # base→tip xyz (metres)
        "distance": float(info["distance"]),
        "is_success": bool(info["is_success"]),
        "episode": episode,
        "step": step,
        "successes": successes,
        "total": total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--viz", dest="viz", action="store_true", default=True,
                    help="launch the live 3D visualizer (default on)")
    ap.add_argument("--no-viz", dest="viz", action="store_false",
                    help="disable the visualizer (terminal-only, original behavior)")
    ap.add_argument("--viz-port", type=int, default=8091)
    ap.add_argument("--camera", type=int, default=None,
                    help="camera index to show (default: pick in the UI)")
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    args = ap.parse_args()

    viz = None
    if args.viz:
        from teleop.rl_reach.viz_server import ReachVizServer
        config = {
            "min_ee": [float(v) for v in MIN_EE],
            "max_ee": [float(v) for v in MAX_EE],
            "success_thresh": float(SUCCESS_THRESH),
        }
        viz = ReachVizServer(config, port=args.viz_port, camera_index=args.camera)
        viz.start()
        time.sleep(1.0)                         # let uvicorn bind before opening the tab
        if not args.no_browser:
            webbrowser.open(viz.url)
        print(f"visualizer: {viz.url}")

    env = SO101ReachEnv()
    model = SAC.load(str(args.model), device=args.device)
    try:
        successes = 0
        for ep in range(args.episodes):
            obs, info = env.reset()
            if viz:
                viz.publish(_make_state(info, ep + 1, 0, successes, args.episodes))
            done = False
            step = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = env.step(action)
                step += 1
                done = terminated or truncated
                if viz:
                    viz.publish(_make_state(info, ep + 1, step,
                                            successes + int(info["is_success"]), args.episodes))
            successes += int(info["is_success"])
            print(f"ep {ep:2d}: final dist {info['distance'] * 1000:5.0f} mm  success={info['is_success']}")
        print(f"\nsuccess {successes}/{args.episodes}")
    finally:
        env.close()
        if viz:
            viz.stop()


if __name__ == "__main__":
    main()
