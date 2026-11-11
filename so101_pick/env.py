"""SO-101 pick environment (PROVIDED, you do not need to edit this file).

Task: approach a 2.5 cm cube, grasp it, and lift it 5 cm off the floor.

What you DO edit (see the TODO markers):
    so101_pick/rewards.py        TODO 1: the reward function
    so101_pick/observations.py   TODO 2: what the policy gets to see
    train.py                     TODO 3: algorithm + hyperparameters

Conventions
-----------
* Control: 20 Hz. Each step the policy outputs 5 numbers in [-1, 1]:
      [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, gripper]
  Each is a *delta* applied to the joint position target, which the
  simulated servo then tracks (position actuators). Wrist roll is frozen.
* The cube spawns on the floor in front of the arm, with its faces lined up
  with the gripper, so a top-down grasp works.
* An episode ends (terminated) on success, or is truncated after
  `max_episode_steps`.
* The reward and observation are computed from a `state` dict (see
  `_get_state`) by YOUR functions in rewards.py / observations.py.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "..", "assets", "so101", "pick_scene.xml")

# --- Task constants -----------------------------------------------------------
CUBE_HALF = 0.0125            # cube half-edge (m); resting centre height
LIFT_HEIGHT = 0.15            # success: cube centre this far above its rest height
PAN_AXIS_XY = (0.0388353, 0.0)  # world xy of the shoulder_pan axis (cube spawns around it)

# --- Control constants --------------------------------------------------------
ARM_STEP = 0.08               # rad per step at |action| = 1 (pan, lift, elbow, wrist_flex)
GRIPPER_STEP = 0.20           # rad per step at |action| = 1
FRAME_SKIP = 10               # 10 x 5 ms physics steps = 50 ms per env step (20 Hz)
HOME_QPOS = np.array([0.0, -0.5, 0.4, 1.5])   # gripper pointing down, ~9 cm above floor
GRIPPER_OPEN = 0.6            # ~6 cm jaw opening at the tips

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
GRIPPER_JOINT = "gripper"
ACTUATED = ARM_JOINTS + ["wrist_roll", GRIPPER_JOINT]


class SO101PickEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 150,
        randomization: float = 1.0,
        reward_fn: Optional[Callable] = None,
        obs_fn: Optional[Callable] = None,
        render_size: tuple[int, int] = (480, 640),
        camera: str = "front",
    ):
        """
        randomization: 0.0 -> cube always at the same spot; 1.0 -> full random
            range. Useful for curriculum (start at 0, then increase).
        reward_fn / obs_fn: default to the ones you write in rewards.py /
            observations.py.
        """
        super().__init__()
        if reward_fn is None:
            from .rewards import compute_reward as reward_fn
        if obs_fn is None:
            from .observations import build_observation as obs_fn
        self.reward_fn, self.obs_fn = reward_fn, obs_fn
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.randomization = float(randomization)
        self.render_size, self.camera = render_size, camera

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        m = self.model

        # ids
        self._ee_site = m.site("gripperframe").id
        self._cube_body = m.body("cube").id
        self._cube_geom = m.geom("cube_geom").id
        self._floor_geom = m.geom("floor").id
        cj = m.joint("cube_joint")
        self._cube_qadr, self._cube_dadr = int(cj.qposadr[0]), int(cj.dofadr[0])
        self._act_ids = [m.actuator(n).id for n in ACTUATED]
        self._act_jnt = [m.joint(n).id for n in ACTUATED]
        self._q_adr = [int(m.jnt_qposadr[j]) for j in self._act_jnt]
        self._v_adr = [int(m.jnt_dofadr[j]) for j in self._act_jnt]
        self._ctrl_lo = m.actuator_ctrlrange[self._act_ids, 0].copy()
        self._ctrl_hi = m.actuator_ctrlrange[self._act_ids, 1].copy()
        # policy action index -> index into the 6 actuators (wrist_roll=4 is frozen)
        self._act_map = [0, 1, 2, 3, 5]

        # jaw geoms = the condim-6 collision geoms of the two gripper bodies
        def jaw_geoms(body: str) -> np.ndarray:
            b = m.body(body).id
            ids = np.where((m.geom_bodyid == b) & (m.geom_condim == 6))[0]
            assert len(ids) > 0, f"no jaw collision geoms found on {body}"
            return ids
        self._fixed_jaw = jaw_geoms("gripper")
        self._moving_jaw = jaw_geoms("moving_jaw_so101_v1")

        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self._renderer = None
        self._steps = 0
        self._target = np.zeros(6)
        self._prev_action = np.zeros(5)
        self._cube_xy0 = np.zeros(2)

        # Observation space is inferred from YOUR obs function.
        self.reset(seed=0)
        probe = np.asarray(self.obs_fn(self._get_state()), dtype=np.float32)
        assert probe.ndim == 1, "build_observation must return a 1-D array"
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=probe.shape, dtype=np.float32)

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        m, d, rng = self.model, self.data, self.np_random
        mujoco.mj_resetData(m, d)

        # arm: home pose with a little noise
        q_arm = HOME_QPOS + rng.uniform(-0.05, 0.05, size=4) * self.randomization
        q_arm = np.clip(q_arm, self._ctrl_lo[:4], self._ctrl_hi[:4])
        self._target = np.array([*q_arm, 0.0, GRIPPER_OPEN])
        for i, adr in enumerate(self._q_adr):
            d.qpos[adr] = self._target[i]
        d.ctrl[self._act_ids] = self._target

        # cube: polar coordinates around the pan axis, faces aligned with the arm
        rs = self.randomization
        radius = 0.18 + rng.uniform(-0.04, 0.04) * rs        # 14 - 22 cm
        angle = rng.uniform(-0.7, 0.7) * rs                  # +-40 degrees
        yaw = angle + rng.uniform(-0.15, 0.15) * rs
        x = PAN_AXIS_XY[0] + radius * np.cos(angle)
        y = PAN_AXIS_XY[1] + radius * np.sin(angle)
        a = self._cube_qadr
        d.qpos[a:a + 3] = [x, y, CUBE_HALF]
        d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self._cube_xy0 = np.array([x, y])

        # let things settle (the cube drops the last ~0 mm; the arm reaches its pose)
        for _ in range(100):
            mujoco.mj_step(m, d)
        self._steps = 0
        self._prev_action = np.zeros(5)
        state = self._get_state()
        return self._obs(state), self._info(state)

    # ------------------------------------------------------------------- step
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(5), -1.0, 1.0)
        self._target[:4] += a[:4] * ARM_STEP
        self._target[5] += a[4] * GRIPPER_STEP
        self._target = np.clip(self._target, self._ctrl_lo, self._ctrl_hi)
        self.data.ctrl[self._act_ids] = self._target
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1

        state = self._get_state(last_action=a)
        reward = float(self.reward_fn(state, a.astype(np.float32)))
        terminated = bool(state["success"])
        truncated = (not terminated) and self._steps >= self.max_episode_steps
        self._prev_action = a
        return self._obs(state), reward, terminated, truncated, self._info(state)

    # ------------------------------------------------------------------ state
    def _get_state(self, last_action: Optional[np.ndarray] = None) -> dict:
        """Everything the reward / observation functions may use (all in metres, radians, SI)."""
        d = self.data
        ee_pos = d.site_xpos[self._ee_site].copy()
        cube_pos = d.xpos[self._cube_body].copy()
        joint_pos = np.array([d.qpos[i] for i in self._q_adr])[self._act_map]
        joint_vel = np.array([d.qvel[i] for i in self._v_adr])[self._act_map]
        ee_to_cube = cube_pos - ee_pos
        cube_height = float(cube_pos[2] - CUBE_HALF)   # 0 when resting on the floor

        # contacts
        n = d.ncon
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        is_cube = (g1 == self._cube_geom) | (g2 == self._cube_geom)
        other = np.where(g1 == self._cube_geom, g2, g1)
        fixed_c = bool(np.any(is_cube & np.isin(other, self._fixed_jaw)))
        moving_c = bool(np.any(is_cube & np.isin(other, self._moving_jaw)))
        is_floor = (g1 == self._floor_geom) | (g2 == self._floor_geom)
        other_f = np.where(g1 == self._floor_geom, g2, g1)
        arm_floor = bool(np.any(is_floor & (other_f != self._cube_geom)))
        is_grasped = fixed_c and moving_c

        return {
            # robot (order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, gripper)
            "joint_pos": joint_pos.astype(np.float32),
            "joint_vel": joint_vel.astype(np.float32),
            "ee_pos": ee_pos.astype(np.float32),                 # between the jaw tips
            "gripper_opening": float(joint_pos[4]),              # rad, ~-0.17 closed .. 1.75 open
            # cube
            "cube_pos": cube_pos.astype(np.float32),
            "cube_quat": d.xquat[self._cube_body].astype(np.float32),
            "cube_lin_vel": d.qvel[self._cube_dadr:self._cube_dadr + 3].astype(np.float32),
            "cube_xy_moved": float(np.linalg.norm(cube_pos[:2] - self._cube_xy0)),
            # relations
            "ee_to_cube": ee_to_cube.astype(np.float32),
            "dist_ee_cube": float(np.linalg.norm(ee_to_cube)),
            "cube_height": cube_height,
            # contacts
            "fixed_jaw_contact": fixed_c,
            "moving_jaw_contact": moving_c,
            "is_grasped": is_grasped,                             # both jaws touching the cube
            "arm_floor_contact": arm_floor,                       # any robot geom touching the floor
            # bookkeeping
            "success": bool(is_grasped and cube_height >= LIFT_HEIGHT),
            "last_action": (self._prev_action if last_action is None else last_action).astype(np.float32),
            "step": self._steps,
        }

    def _obs(self, state: dict) -> np.ndarray:
        return np.asarray(self.obs_fn(state), dtype=np.float32)

    def _info(self, state: dict) -> dict:
        return {
            "is_success": state["success"],          # SB3 reads this key for success rate
            "dist_ee_cube": state["dist_ee_cube"],
            "cube_height": state["cube_height"],
            "is_grasped": state["is_grasped"],
        }

    # ----------------------------------------------------------------- render
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            h, w = self.render_size
            self._renderer = mujoco.Renderer(self.model, h, w)
        self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
