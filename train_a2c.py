from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from make_envs import make_train_env

TOTAL_TIMESTEPS = 600_000  # comparable to PPO but typically needs fewer
N_ENVS = 1                # using a single environment as requested


def make_vec_env(version: str):
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
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
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
        run_id = None
    elif len(sys.argv) == 2:
        version = sys.argv[1]
        run_id = None
    else:
        version = sys.argv[1]
        run_id = int(sys.argv[2])
    main(version, run_id)