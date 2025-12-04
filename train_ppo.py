# train_ppo.py
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from make_envs import make_train_env, make_train_env_new

N_ENVS = 2         # number of parallel training environments
TOTAL_TIMESTEPS = 800_000  # can bump to 1-2M later as needed


def env_chooser(version):
    if version == "new":
        def _make_env():
            return make_train_env_new()
    else:
        def _make_env():
            return make_train_env()
    return _make_env


def main(version):       
    # Wrap in VecEnv for SB3
    vec_env = DummyVecEnv([env_chooser(version) for _ in range(N_ENVS)])

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    # Define and train the PPO model
    model = PPO(
        "MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size = 64,
        n_steps=32,
        ent_coef=0.02,
    )

    # Learn for 300,000 timesteps where each timestep is one env step
    # In the env step, we move forward by one day
    # since we have 1400 days of data, this is about 214 epochs
    # (total_timesteps = epochs * train_T)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    vec_env.save("vecnormalize_stats_ppo.pkl")
    model.save("crypto_portfolio_ppo.zip")
    print("Saved trained model to ppo_crypto_portfolio.zip")

if __name__ == "__main__":
    print("Usage: python train_ppo.py [original|new]")
    import sys 
    if len(sys.argv) > 1 and sys.argv[1] == "new":
        main("new")
    else: 
        main("original")