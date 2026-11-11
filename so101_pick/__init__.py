"""SO-101 pick lab: a MuJoCo + Gymnasium environment for RL with Stable-Baselines3."""
import gymnasium as gym

from .env import SO101PickEnv

gym.register(
    id="SO101Pick-v0",
    entry_point="so101_pick.env:SO101PickEnv",
    max_episode_steps=150,
)

__all__ = ["SO101PickEnv"]
