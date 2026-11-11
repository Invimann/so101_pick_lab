"""Self-check for TODO 1 (your reward). These tests use synthetic states, no training needed.
They FAIL with the starter reward and should PASS once your staged reward is sensible.
    python -m pytest tests/test_reward.py -v      (or)      python tests/test_reward.py
Passing is necessary, not sufficient: a good reward must still *train*.
"""
import copy, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from so101_pick import SO101PickEnv
from so101_pick.rewards import compute_reward

ZERO = np.zeros(5, dtype=np.float32)


def base_state():
    env = SO101PickEnv(); env.reset(seed=0)
    return env._get_state()


def with_dist(s, dist):
    """Place the jaws `dist` metres straight above the cube."""
    s = copy.deepcopy(s)
    s["ee_pos"] = s["cube_pos"] + np.array([0, 0, dist], dtype=np.float32)
    s["ee_to_cube"] = s["cube_pos"] - s["ee_pos"]
    s["dist_ee_cube"] = float(dist)
    return s


def grasped(s, height):
    s = with_dist(s, 0.005)
    s.update(fixed_jaw_contact=True, moving_jaw_contact=True, is_grasped=True,
             cube_height=float(height), gripper_opening=0.05)
    s["cube_pos"] = s["cube_pos"].copy(); s["cube_pos"][2] = 0.0125 + height
    s["success"] = height >= 0.05
    return s


def test_reward_is_finite_float():
    r = compute_reward(base_state(), ZERO)
    assert np.isfinite(r)


def test_getting_closer_is_rewarded():
    s = base_state()
    assert compute_reward(with_dist(s, 0.02), ZERO) > compute_reward(with_dist(s, 0.10), ZERO)


def test_grasping_beats_hovering():
    s = base_state()
    assert compute_reward(grasped(s, 0.0), ZERO) > compute_reward(with_dist(s, 0.005), ZERO)


def test_lifting_while_grasped_is_rewarded():
    s = base_state()
    assert compute_reward(grasped(s, 0.03), ZERO) > compute_reward(grasped(s, 0.0), ZERO)


def test_success_is_the_best_outcome():
    s = base_state()
    best_non_success = compute_reward(grasped(s, 0.045), ZERO)
    assert compute_reward(grasped(s, 0.05), ZERO) > best_non_success


def test_closing_gripper_in_empty_air_is_not_rewarded():
    s = with_dist(base_state(), 0.10)
    closed = copy.deepcopy(s); closed["gripper_opening"] = -0.1
    opened = copy.deepcopy(s); opened["gripper_opening"] = 0.6
    assert compute_reward(closed, ZERO) <= compute_reward(opened, ZERO) + 1e-6


def test_lifting_without_grasp_is_not_rewarded():
    """Cube 3 cm up but NOT grasped (e.g. thrown/balanced) must not look like progress."""
    s = base_state()
    g = grasped(s, 0.03)
    air = copy.deepcopy(g); air.update(is_grasped=False, fixed_jaw_contact=False, moving_jaw_contact=False)
    assert compute_reward(air, ZERO) < compute_reward(g, ZERO)


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for name, t in tests:
        try:
            t(); print("PASS", name)
        except AssertionError:
            failed += 1; print("FAIL", name)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
