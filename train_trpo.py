from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from make_envs import make_train_env

TOTAL_TIMESTEPS = 800_000
N_ENVS = 1

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

    model = TRPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=1e-4,
        target_kl=0.025,
        seed=run_id if run_id is not None else 0,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    if run_id is None:
        model_path = "crypto_portfolio_trpo.zip"
        vecnorm_path = "vecnormalize_stats_trpo.pkl"
    else:
        model_path = f"crypto_portfolio_trpo_run{run_id}.zip"
        vecnorm_path = f"vecnormalize_stats_trpo_run{run_id}.pkl"
    model.save(f"./models/{model_path}")
    vec_env.save(f"./models/{vecnorm_path}")
    print(f"Saved TRPO model to {model_path}")
    print(f"Saved VecNormalize stats to {vecnorm_path}")

if __name__ == "__main__":
    print("Usage: python train_trpo.py [original|new]")
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