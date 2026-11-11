"""TODO 1: the reward function  (this is the heart of the lab).

The task has three phases: APPROACH the cube, GRASP it, LIFT it.
With the starter reward below (a single sparse bonus on success), the policy
almost never sees any reward, so it learns nothing. Your job is to shape the
reward so that RL can discover the whole behaviour.

`state` is a dict (all SI units). Keys you can use:

    ee_pos            (3,)  position between the jaw tips
    cube_pos          (3,)  cube centre
    ee_to_cube        (3,)  cube_pos - ee_pos
    dist_ee_cube      float distance between jaw tips and cube centre (m)
    gripper_opening   float jaw joint angle (rad): ~-0.17 closed ... 1.75 wide open
                            (the cube is 2.5 cm wide: ~0.3 rad is "just wider than the cube")
    cube_height       float cube centre height above its resting height (m); 0 on the floor
    cube_xy_moved     float how far the cube slid sideways since reset (m)
    fixed_jaw_contact bool  fixed jaw touches the cube
    moving_jaw_contact bool moving jaw touches the cube
    is_grasped        bool  BOTH jaws touch the cube
    arm_floor_contact bool  some part of the robot touches the floor
    joint_vel         (5,)  joint velocities (rad/s)
    success           bool  grasped AND cube_height >= 5 cm (episode ends)
    last_action       (5,)  the action just applied, in [-1, 1]
    step              int   steps taken so far in this episode: 0 at reset, then 1, 2, ...
                            An episode is at most 150 steps (20 Hz = 7.5 s), so
                            `150 - state["step"]` is the number of steps left.
                            Use it to encourage quickness (see the design questions).

`action` is the same 5-vector in [-1, 1].

Design questions to think about:
  * Why is a reward based on distance (dense) easier to learn from than a
    single success bonus (sparse)?
  * How do you make the reward "switch on" the next phase only when the
    previous one is done? (e.g. only reward lifting once the cube is grasped.)
  * Can the policy cheat? Think about: knocking the cube away, hovering next to
    the cube forever, slamming the arm into the floor, jittery actions.
  * Scale matters: keep each term at a similar magnitude (roughly <= 1 per step),
    and make success clearly the best outcome.
  * Quickness: success ends the episode. How can you make finishing sooner better
    than finishing later? (e.g. a small cost per step, or a success bonus that grows
    with the number of steps left.) Beware: if the per-step rewards you collect while
    holding the cube add up to more than the success bonus, the policy can learn to
    stall just below the success height.

Check your work with:  python -m pytest tests/test_reward.py   (or: python tests/test_reward.py)
"""
import numpy as np


def compute_reward(state: dict, action: np.ndarray) -> float:
    # ----------------------------------------------------------------------
    # TODO 1: replace this starter with a staged, dense reward.
    #
    # Suggested structure (you decide the exact terms and weights):
    #   1. approach: reward for reducing the distance between jaws and cube
    #   2. grasp:    reward for closing the gripper *around* the cube
    #                (use is_grasped; avoid rewarding a closed gripper in empty air)
    #   3. lift:     reward for cube_height, only while grasped
    #   4. success:  a large bonus when state["success"]
    #   + small penalties if you need them (floor contact, big actions, ...)
    # ----------------------------------------------------------------------
    return 1.0 if state["success"] else 0.0
