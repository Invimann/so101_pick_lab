"""Tests for the provided environment. These should all pass out of the box.
    python -m pytest tests/test_env.py        (or)        python tests/test_env.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from so101_pick import SO101PickEnv


def test_reset_is_seeded():
    e = SO101PickEnv()
    o1, _ = e.reset(seed=3); o2, _ = e.reset(seed=3)
    assert np.allclose(o1, o2)


def test_observation_matches_space():
    e = SO101PickEnv()
    o, _ = e.reset(seed=0)
    assert o.dtype == np.float32 and e.observation_space.contains(o)
    for _ in range(20):
        o, *_ = e.step(e.action_space.sample())
        assert o.shape == e.observation_space.shape and np.all(np.isfinite(o))


def test_zero_action_keeps_cube_still():
    e = SO101PickEnv(); e.reset(seed=0)
    p0 = e.data.xpos[e._cube_body].copy()
    for _ in range(30):
        e.step(np.zeros(5))
    assert np.linalg.norm(e.data.xpos[e._cube_body] - p0) < 0.01


def test_no_randomization_is_fixed_layout():
    e = SO101PickEnv(randomization=0.0)
    e.reset(seed=1); a = e.data.xpos[e._cube_body].copy()
    e.reset(seed=2); b = e.data.xpos[e._cube_body].copy()
    assert np.allclose(a, b, atol=1e-3)


def test_actions_are_clipped():
    e = SO101PickEnv(); e.reset(seed=0)
    e.step(np.full(5, 100.0))
    assert np.all(e._target >= e._ctrl_lo - 1e-9) and np.all(e._target <= e._ctrl_hi + 1e-9)


def test_truncation():
    e = SO101PickEnv(max_episode_steps=5); e.reset(seed=0)
    for i in range(5):
        _, _, term, trunc, _ = e.step(np.zeros(5))
    assert trunc and not term


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(); print("PASS", t.__name__)
    print(f"{len(tests)} passed")
