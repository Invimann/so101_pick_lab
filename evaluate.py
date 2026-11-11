"""Evaluate a trained policy: success rate over N episodes, optionally record a video.
(PROVIDED; this is also how your result is graded.)

    python evaluate.py runs/first_try/best_model.zip --episodes 100
    python evaluate.py runs/first_try/best_model.zip --episodes 5 --video rollout.mp4
    (add --vecnormalize runs/first_try/vecnormalize.pkl if you trained with VecNormalize)
"""
import argparse

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import so101_pick  # noqa: F401


def load(path):
    for cls in (SAC, PPO):
        try:
            return cls.load(path, device="cpu")
        except Exception:
            continue
    raise RuntimeError(f"Could not load {path} as SAC or PPO")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--randomization", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--video", default=None, help="write the first few episodes to this .mp4")
    ap.add_argument("--vecnormalize", default=None)
    args = ap.parse_args()

    render = args.video is not None
    venv = DummyVecEnv([lambda: gym.make("SO101Pick-v0", randomization=args.randomization,
                                         render_mode="rgb_array" if render else None)])
    if args.vecnormalize:
        venv = VecNormalize.load(args.vecnormalize, venv)
        venv.training = False
        venv.norm_reward = False
    model = load(args.model)
    venv.seed(args.seed)

    successes, grasped, frames = 0, 0, []
    for ep in range(args.episodes):
        obs = venv.reset()
        done, ever_grasped = False, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_arr, infos = venv.step(action)
            done = bool(done_arr[0])
            info = infos[0]
            ever_grasped |= bool(info.get("is_grasped", False))
            if render and ep < 5:
                frames.append(venv.envs[0].unwrapped.render())
        successes += int(info.get("is_success", False))
        grasped += int(ever_grasped)

    print(f"episodes: {args.episodes}")
    print(f"success rate (grasped + lifted 5 cm): {successes / args.episodes:.1%}")
    print(f"grasped at least once:                {grasped / args.episodes:.1%}")
    if render and frames:
        imageio.mimsave(args.video, frames, fps=20)
        print("video written to", args.video)


if __name__ == "__main__":
    main()
