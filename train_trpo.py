"""
train_trpo.py
Train a TRPO agent on the original MDP environment.

TRPO is a trust-region policy optimization method that ensures very stable
policy updates under KL-divergence constraints.

Literature support:
- Schulman et al. (2015) – TRPO original paper.
- Zhang, Zohren & Roberts (2020) – Use TRPO variants in financial RL.
- TRPO is known to produce smooth, safe updates beneficial in noisy markets.

"""

from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from make_envs import make_train_env, make_train_env_new

TOTAL_TIMESTEPS = 600_000
N_ENVS = 1

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

    # TRPO-specific literature-guided hyperparameters
    model = TRPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=1e-4,     # TRPO uses natural gradients; LR kept small
        n_steps=2048,           # TRPO prefers long rollouts for stable CG estimation
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model.save("crypto_portfolio_trpo.zip")
    vec_env.save("vecnormalize_stats_trpo.pkl")

    print("Saved TRPO model and VecNormalize stats.")


if __name__ == "__main__":
    print("Usage: python train_trpo.py [original|new]")
    import sys 
    if len(sys.argv) > 1 and sys.argv[1] == "new":
        main("new")
    else: 
        main("original")