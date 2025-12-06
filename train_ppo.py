# train_ppo.py
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from make_envs import make_train_env

N_ENVS = 1         # number of parallel training environments
TOTAL_TIMESTEPS = 800_000  # can bump to 1-2M later as needed

def make_vec_env(version):
    def _init():
         return make_train_env(version)
    vec_env = DummyVecEnv([_init for _ in range(N_ENVS)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )
    return vec_env

def main(version: str, run_id=None):
    vec_env = make_vec_env(version)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        gamma=0.99,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        ent_coef=0.01,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    if run_id is None:
        model_path = "crypto_portfolio_ppo.zip"
        vecnorm_path = "vecnormalize_stats_ppo.pkl"
    else:
        model_path = f"crypto_portfolio_ppo_run{run_id}.zip"
        vecnorm_path = f"vecnormalize_stats_ppo_run{run_id}.pkl"
    model.save(f"./models/{model_path}")
    vec_env.save(f"./models/{vecnorm_path}")
    print(f"Saved PPO model to {model_path}")
    print(f"Saved VecNormalize stats to {vecnorm_path}")

if __name__ == "__main__":
    print("Usage: python train_ppo.py [original|new]")
    import sys
    if len(sys.argv) == 1:
        version = "new"
        run_id = None
    elif len(sys.argv) == 2:
        version = sys.argv[1]
        run_id = None
    else:
        version = sys.argv[1]
        run_id = int(sys.argv[2])
    main(version, run_id)