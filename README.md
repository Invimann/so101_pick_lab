# Lab: RL pick with the SO-101 arm (MuJoCo + Stable-Baselines3)

Teach a simulated SO-101 arm to **approach, grasp and lift** a 2.5 cm cube using
reinforcement learning. No ROS 2 needed: just Python, MuJoCo, Gymnasium and SB3.

**Setup:** follow `installation.md`, then run `python scripts/check_env.py`.

## What is provided vs. what you write

| File | Who | Purpose |
|---|---|---|
| `assets/so101/` | provided | SO-101 MuJoCo model (MuJoCo Menagerie, Apache-2.0) + `pick_scene.xml` (floor + cube) |
| `so101_pick/env.py` | provided | Gymnasium env: physics, actions, contacts, resets, success check |
| `so101_pick/rewards.py` | **YOU (TODO 1)** | the reward function |
| `so101_pick/observations.py` | provided | what the policy sees (21-D, documented in the file) |
| `train.py` | **YOU (TODO 2)** + provided harness | tune the baseline hyperparameters (SAC or PPO) |
| `evaluate.py` | provided | success rate over N episodes, optional video |
| `tests/` | provided | `test_env.py` (should pass), `test_reward.py` (checks your TODO 1) |

## The task in one paragraph
Each step the policy outputs 5 numbers in [-1, 1]: change of shoulder pan, shoulder lift, elbow,
wrist flex, and gripper (20 Hz, 150 steps per episode). The cube appears on the floor in front of
the arm. **Success** = both jaws touch the cube *and* the cube centre is >= 5 cm above the floor.

## Workflow
1. `python scripts/check_env.py` and `python tests/test_env.py`: everything passes.
2. **Part A:** keep the starter reward and the baseline `HPARAMS` in `train.py`, and train briefly
   with `--randomization 0`. Look at TensorBoard: why does it not learn?
3. **Part B (TODO 1):** write the staged reward. Run `python tests/test_reward.py`, then train with
   `--randomization 0` (cube always in the same place).
4. **Part C (TODO 2):** tune the baseline hyperparameters (one change per run, different `--run-name`),
   then train with `--randomization 1` (random cube positions).
5. Evaluate: `python evaluate.py runs/<name>/best_model.zip --episodes 100 --video rollout.mp4`

```bash
python train.py --run-name my_run --timesteps 200000 --randomization 0
tensorboard --logdir runs            # Colab: %load_ext tensorboard ; %tensorboard --logdir runs
```

## Report questions
* What did the policy do with the starter reward, and why?
* Which reward terms mattered? Show one failure you saw (e.g. pushing the cube, hovering) and how you fixed it.
* Hyperparameters: which values did you change from the baseline, why, and what happened (include a plot)?
* Why did you choose your algorithm (SAC or PPO)?
* Final success rate over 100 episodes with random cube positions.
