"""
train_a2c.py

Training script for A2C-based agents on the crypto portfolio environment.

Purpose:
- Construct a VecNormalize-wrapped training environment using the project's
  make_train_env factory.
- Instantiate and train a Stable-Baselines3 A2C agent with sensible defaults
  for financial time-series (gamma close to 1, modest learning rate).
- Save the trained model and VecNormalize statistics for consistent evaluation.

Notes:
- This file intentionally preserves the original command-line interface and
  file paths used by CI/tests. Only documentation and comments were added.
"""
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from make_envs import make_train_env

TOTAL_TIMESTEPS = 800_000  # comparable to PPO but typically needs fewer
N_ENVS = 1                # using a single environment


def make_vec_env(version: str):
    """
    Create and return a VecNormalize-wrapped vectorized training environment.

    Parameters
    ----------
    version : str
        Environment version string passed to make_train_env (e.g. "new" or "original").

    Returns
    -------
    stable_baselines3.common.vec_env.VecNormalize
        A DummyVecEnv wrapped by VecNormalize for training.
    """
    def _init():
         return make_train_env(version)
    # Build a single-worker DummyVecEnv and wrap with VecNormalize so agents see
    # normalized observations and rewards during training.
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
    Train an A2C agent on the specified environment version and save artifacts.

    Parameters
    ----------
    version : str
        Environment version ("new" or "original").
    run_id : int
        Run identifier used to name saved artifacts.
    exp_id : int
        Experiment identifier used to seed randomness.
    """
    seed = exp_id * 100 + run_id
    print(f"[A2C] version={version}, exp_id={exp_id}, run_id={run_id}, seed={seed}")

    # Ensure deterministic seeding for reproducibility across runs
    set_random_seed(seed)
    vec_env = make_vec_env(version)
    
    # Instantiate the A2C model with finance-appropriate hyperparameters
    model = A2C(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=7e-4,       # SB3 default; A2C tends to like slightly higher LR
        gamma=0.99,               # long-term reward focus (finance-appropriate)
        n_steps=5,                # SB3 default; A2C uses small rollout steps
        gae_lambda=0.95,          # standard for advantage estimation
        ent_coef=0.0,             # entropy often not needed for A2C; can set to 0.01 if you want exploration
        vf_coef=0.5,              # value loss coefficient (default)
        seed=seed,
    )

    # Train the model for the configured number of timesteps
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # Persist model and VecNormalize statistics using the same naming conventions
    if run_id is None:
        model_path = "crypto_portfolio_a2c.zip"
        vecnorm_path = "vecnormalize_stats_a2c.pkl"
    else:
        model_path = f"crypto_portfolio_a2c_run{run_id}.zip"
        vecnorm_path = f"vecnormalize_stats_a2c_run{run_id}.pkl"
    model.save(f"./models/{model_path}")
    vec_env.save(f"./models/{vecnorm_path}")
    print(f"Saved A2C model to {model_path}")
    print(f"Saved VecNormalize stats to {vecnorm_path}")

if __name__ == "__main__":
    print("Usage: python train_a2c.py [original|new]")
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