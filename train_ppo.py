"""
train_ppo.py

Training script for sequence-aware PPO (RecurrentPPO) on the crypto portfolio env.

Purpose:
- Build a VecNormalize-wrapped training environment using make_train_env.
- Instantiate and train a RecurrentPPO (LSTM) policy so the agent has temporal
  memory across timesteps.
- Save the trained model and VecNormalize running statistics for later evaluation.

Notes:
- This change only adds documentation and lightweight comments; it does not
  modify runtime behaviour or function signatures.
"""
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from make_envs import make_train_env

N_ENVS = 1         # number of parallel training environments
TOTAL_TIMESTEPS = 800_000  # can bump to 1-2M later as needed

def make_vec_env(version):
    """
    Create a VecNormalize-wrapped vectorized training environment.

    Parameters
    ----------
    version : str
        Environment version passed to make_train_env (e.g., "new" or "original").

    Returns
    -------
    VecNormalize-wrapped DummyVecEnv
        Environment ready for training (observations/rewards normalized).
    """
    def _init():
         return make_train_env(version)
    # Build a DummyVecEnv with N_ENVS workers (here typically 1) and wrap it
    # with VecNormalize so training sees normalized observations & rewards.
    vec_env = DummyVecEnv([_init for _ in range(N_ENVS)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )
    return vec_env

def main(version: str, run_id: int = 0, exp_id: int = 0):
    """
    Train a RecurrentPPO (LSTM) agent and persist model + VecNormalize stats.

    Parameters
    ----------
    version : str
        Environment version ("new" or "original").
    run_id : int
        Run identifier used to name saved artifacts.
    exp_id : int
        Experiment identifier used to derive a reproducible seed.
    """
    seed = exp_id * 100 + run_id
    print(f"[RecurrentPPO] version={version}, exp_id={exp_id}, run_id={run_id}, seed={seed}")

    # Ensure deterministic seeding for reproducibility
    set_random_seed(seed)
    vec_env = make_vec_env(version)

    # Instantiate RecurrentPPO with an LSTM policy so the model receives and
    # maintains hidden state across timesteps (sequence-aware).
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=vec_env,
        verbose=1,
        n_steps=128,      # shorter rollouts suitable for recurrent updates
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        seed=seed,
    )

    # Run learning loop for configured timesteps
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    # Save model and VecNormalize stats using established naming convention
    if run_id is None:
        model_path = "crypto_portfolio_ppo.zip"
        vecnorm_path = "vecnormalize_stats_ppo.pkl"
    else:
        model_path = f"crypto_portfolio_ppo_run{run_id}.zip"
        vecnorm_path = f"vecnormalize_stats_ppo_run{run_id}.pkl"
    model.save(f"./models/{model_path}")
    vec_env.save(f"./models/{vecnorm_path}")
    print(f"Saved RecurrentPPO model to {model_path}")
    print(f"Saved VecNormalize stats to {vecnorm_path}")

if __name__ == "__main__":
    print("Usage: python train_ppo.py [original|new]")
    import sys
    if len(sys.argv) == 1:
        version = "new"
        run_id = 0
        exp_id = 0
    elif len(sys.argv) == 2:
        version = sys.argv[1]
        run_id = 0
        exp_id = 0
    elif len(sys.argv) == 3:
        version = sys.argv[1]
        run_id = int(sys.argv[2])
        exp_id = 0
    else:
        version = sys.argv[1]
        run_id = int(sys.argv[2])
        exp_id = int(sys.argv[3])
    main(version, run_id, exp_id)