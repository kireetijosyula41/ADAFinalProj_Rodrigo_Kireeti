"""
train_a2c.py
Train an A2C agent on the original MDP environment.

A2C (Advantage Actor-Critic) is a synchronous advantage-based method often
used in financial RL because of its stability and simplicity.

Literature support:
- Moody & Saffell (2001) – Reinforcement learning trading via actor-critic.
- Deng et al. (2016) – Deep RL for stock trading uses actor-critic variants.
- SB3 documentation – A2C is stable for single-env settings.

"""

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from make_envs import make_train_env, make_train_env_new

TOTAL_TIMESTEPS = 600_000  # comparable to PPO but typically needs fewer
N_ENVS = 1                # using a single environment as requested


def env_chooser(version):
    if version == "new":
        def _make_env():
            return make_train_env_new()
    else:
        def _make_env():
            return make_train_env()
    return _make_env

def main(version):
    vec_env = DummyVecEnv([env_chooser(version) for _ in range(N_ENVS)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    # Financial RL literature often uses slightly higher learning rates for AC methods.
    model = A2C(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=7e-4,       # SB3 default; A2C tends to like slightly higher LR
        gamma=0.99,               # long-term reward focus (finance-appropriate)
        n_steps=5,                # SB3 default; A2C uses small rollout steps
        gae_lambda=0.95,          # standard for advantage estimation
        ent_coef=0.0,             # entropy often not needed for A2C; can set to 0.01 if you want exploration
        vf_coef=0.5               # value loss coefficient (default)
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # Save model + normalization stats
    model.save("crypto_portfolio_a2c.zip")
    vec_env.save("vecnormalize_stats_a2c.pkl")

    print("Saved A2C model and VecNormalize stats.")


if __name__ == "__main__":
    print("Usage: python train_a2c.py [original|new]")
    import sys 
    if len(sys.argv) > 1 and sys.argv[1] == "new":
        main("new")
    else: 
        main("original")