"""What the policy sees (PROVIDED: you do not need to edit this file).

The policy is a neural network that only knows the vector returned here. It is
a 21-number vector, built from the `state` dict (same keys as in rewards.py):

    joint_pos     (5,)  shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, gripper (rad)
    joint_vel     (5,)  the same joints (rad/s), multiplied by 0.1 to keep magnitudes ~ 1
    ee_pos        (3,)  position between the jaw tips (m)
    cube_pos      (3,)  cube centre (m)
    ee_to_cube    (3,)  cube_pos - ee_pos: a "relative" feature that tells the policy
                        directly which way to move the gripper
    contact flags (2,)  fixed_jaw_contact, moving_jaw_contact as 0.0 / 1.0

Why these choices:
  * joint angles + velocities describe the arm; ee_pos spares the network from
    learning the arm's forward kinematics;
  * cube_pos and ee_to_cube say where the target is, in absolute and relative terms;
  * the contact flags let the policy "feel" whether it is holding the cube.

A note on realism: cube pose (from a camera/tracker) and joint states are easy to
measure on a real SO-101. The contact flags are not free on hardware (you would need
touch sensing or motor-current thresholds), so keep that in mind when discussing
sim-to-real in your report.

Heads up: a saved model is tied to this observation size (21). Do not change it
after training, or the checkpoint can no longer be loaded.
"""
import numpy as np


def build_observation(state: dict) -> np.ndarray:
    return np.concatenate([
        state["joint_pos"],                       # 5
        0.1 * state["joint_vel"],                 # 5 (scaled)
        state["ee_pos"],                          # 3
        state["cube_pos"],                        # 3
        state["ee_to_cube"],                      # 3
        [float(state["fixed_jaw_contact"]), float(state["moving_jaw_contact"])],  # 2
    ]).astype(np.float32)
