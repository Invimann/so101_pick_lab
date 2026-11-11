"""Training script (PROVIDED harness + TODO: tune the hyperparameters).

Usage examples
--------------
    python train.py --run-name first_try --timesteps 200000
    python train.py --run-name finetune --init checkpoints/warmstart.zip --timesteps 100000
    tensorboard --logdir runs        # in another terminal / Colab cell

Outputs: runs/<run-name>/  (TensorBoard logs, best_model.zip, final_model.zip, vecnormalize.pkl)

Your workflow here: run the BASELINE below once, look at the learning curves in
TensorBoard, then change ONE hyperparameter at a time and compare runs
(use a different --run-name for each experiment).
"""
import argparse
import os

import gymnasium as gym
from stable_baselines3 import PPO, SAC  # noqa: F401  (both are available to you)
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

import so101_pick  # noqa: F401  (registers "SO101Pick-v0")

# =============================================================================
# TODO 3a: choose the algorithm.  Set ALGO to "SAC" or "PPO".
#   SAC: off-policy, usually needs far fewer environment steps. 1 env is fine.
#   PPO: on-policy, needs many parallel envs (set N_ENVS below) and many more steps.
# The default is "SAC". In your report: justify your choice in 2-3 sentences.
# =============================================================================
ALGO = "SAC"  # <- "SAC" or "PPO"

# =============================================================================
# TODO 3b: BASELINE hyperparameters: a reasonable starting point, NOT a tuned one.
# It may learn slowly or not at all. Your job is to improve it: change one value (or few) at a
# time, re-run, and compare the learning curves (success rate, episode reward).
# Things worth testing:
#   SAC: learning_rate, batch_size, buffer_size, gamma, tau, ent_coef ("auto" or a fixed
#        value such as 0.05), learning_starts, train_freq / gradient_steps, net_arch
#   PPO: learning_rate, n_steps (per env!), batch_size, n_epochs, gamma, gae_lambda,
#        clip_range, ent_coef, net_arch
# Docs: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html  (and ppo.html)
# In your report: say what you changed, why, and what happened (with a plot).
# =============================================================================
HPARAMS_SAC = {
    "learning_rate": 3e-4,
    "batch_size": 256,
    "buffer_size": 100_000,
    "learning_starts": 1_000,
    "gamma": 0.99,
    "tau": 0.005,
    "ent_coef": "auto",
    "train_freq": 1,
    "gradient_steps": 1,
    "policy_kwargs": dict(net_arch=[256, 256]),
}

HPARAMS_PPO = {
    "learning_rate": 3e-4,
    "n_steps": 2048,          # steps per env per update
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "policy_kwargs": dict(net_arch=[64, 64]),
}

HPARAMS = HPARAMS_SAC if ALGO == "SAC" else HPARAMS_PPO

# TODO 3c (optional): normalise observations/rewards with VecNormalize? Set True/False and say why.
USE_VECNORMALIZE = False

N_ENVS = 4  # parallel environments. SAC: 1-4 is fine. PPO: try 8+ (more CPU cores help).


def make_env(randomization: float):
    return lambda: gym.make("SO101Pick-v0", randomization=randomization)


def build_model(venv, tb_dir: str):
    if ALGO not in ("SAC", "PPO"):
        raise ValueError("ALGO must be 'SAC' or 'PPO' at the top of train.py")
    cls = {"SAC": SAC, "PPO": PPO}[ALGO]
    return cls("MlpPolicy", venv, tensorboard_log=tb_dir, verbose=1, **HPARAMS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--randomization", type=float, default=1.0,
                    help="0 = cube always at the same place, 1 = fully random (start low, then raise)")
    ap.add_argument("--init", default=None, help="path to a .zip checkpoint to fine-tune from")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = os.path.join("runs", args.run_name)
    os.makedirs(out, exist_ok=True)

    vec_cls = SubprocVecEnv if N_ENVS > 1 else DummyVecEnv
    venv = make_vec_env(make_env(args.randomization), n_envs=N_ENVS, seed=args.seed, vec_env_cls=vec_cls)
    eval_env = make_vec_env(make_env(args.randomization), n_envs=1, seed=args.seed + 1000)
    if USE_VECNORMALIZE:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True)
        eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    if args.init:
        cls = {"SAC": SAC, "PPO": PPO}[ALGO]
        model = cls.load(args.init, env=venv, tensorboard_log=out, **{k: v for k, v in HPARAMS.items()
                                                                       if k in ("learning_rate",)})
        print(f"Loaded checkpoint {args.init}")
    else:
        model = build_model(venv, out)

    eval_cb = EvalCallback(eval_env, best_model_save_path=out, log_path=out,
                           eval_freq=max(10_000 // N_ENVS, 1), n_eval_episodes=20, deterministic=True)
    model.learn(total_timesteps=args.timesteps, callback=CallbackList([eval_cb]),
                tb_log_name="tb", progress_bar=False)
    model.save(os.path.join(out, "final_model"))
    if USE_VECNORMALIZE:
        venv.save(os.path.join(out, "vecnormalize.pkl"))
    print("Done. Models saved in", out)


if __name__ == "__main__":
    main()
