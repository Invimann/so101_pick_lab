"""Smoke test for the lab environment. Run from the repo root:

    python scripts/check_env.py            # full check incl. offscreen rendering
    python scripts/check_env.py --no-render   # skip rendering (e.g. no GL available yet)

Prints "All checks passed" when everything works.
"""
import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def step(msg):
    print(f"[ .. ] {msg}", end="", flush=True)


def ok(extra=""):
    print(f"\r[ OK ] {extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    step("imports (mujoco, gymnasium, stable-baselines3, torch)")
    import gymnasium as gym
    import mujoco
    import numpy as np
    import stable_baselines3
    import torch
    ok(f"mujoco {mujoco.__version__}, gymnasium {gym.__version__}, "
       f"sb3 {stable_baselines3.__version__}, torch {torch.__version__}")

    step("importing so101_pick and loading the SO-101 model")
    import so101_pick  # noqa: F401
    env = gym.make("SO101Pick-v0", render_mode=None if args.no_render else "rgb_array")
    base = env.unwrapped
    ok(f"model loaded: {base.model.nbody} bodies, {base.model.nu} actuators")

    step("reset + observation")
    obs, info = env.reset(seed=0)
    assert np.all(np.isfinite(obs)), "observation has NaN/inf"
    assert env.observation_space.contains(obs), "observation not inside observation_space"
    ok(f"observation shape {obs.shape}, action shape {env.action_space.shape}")

    step("300 random steps")
    t0 = time.time()
    n = 0
    for _ in range(300):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        n += 1
        assert np.all(np.isfinite(obs)) and np.isfinite(r), "NaN/inf in step output"
        if term or trunc:
            env.reset()
    ok(f"{n / (time.time() - t0):.0f} steps/s")

    step("Stable-Baselines3 env checker")
    from stable_baselines3.common.env_checker import check_env
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(env.unwrapped, warn=False, skip_render_check=True)
    ok()

    if not args.no_render:
        step("offscreen render")
        env.reset(seed=1)
        img = env.render()
        assert img is not None and img.ndim == 3 and img.any(), "render returned an empty image"
        ok(f"frame {img.shape}")
        try:
            import imageio
            imageio.imwrite("check_env_frame.png", img)
            print("       saved check_env_frame.png: open it and you should see the arm and a red cube")
        except Exception as e:  # noqa: BLE001
            print("       (could not save PNG:", e, ")")

    env.close()
    print("\nAll checks passed")


if __name__ == "__main__":
    main()
