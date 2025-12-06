import sys
import numpy as np
import torch

from stable_baselines3 import PPO, A2C
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from make_envs import make_test_env

ALGOS = {
    "ppo": PPO,
    "a2c": A2C,
    "trpo": TRPO,
}

def make_eval_vec_env(version: str, algo: str):
    def _init():
        return make_test_env(version)
    vec_env = DummyVecEnv([_init])

    stats_path = f"vecnormalize_stats_{version}_{algo}.pkl"
    vec_env = VecNormalize.load(stats_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env

def ensemble_action(models, obs):
    """
    Soft voting: average action probabilities over models.
    obs: np.ndarray of shape (1, obs_dim)
    """
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

    probs_list = []
    for m in models:
        with torch.no_grad():
            dist = m.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy()[0]
            probs_list.append(probs)

    mean_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    mean_probs = mean_probs / np.sum(mean_probs)  # normalize
    action = int(np.argmax(mean_probs))
    return action, mean_probs

def evaluate_ensemble(algo: str, version: str, run_ids):
    ModelClass = ALGOS[algo]

    # 1. Make eval env with proper normalization
    vec_env = make_eval_vec_env(version, algo)

    # 2. Load all models
    models = []
    for r in run_ids:
        model_path = f"{algo}_{version}_run{r}.zip"
        model = ModelClass.load(model_path, env=vec_env)
        models.append(model)

    # 3. Roll out ensemble policy
    obs = vec_env.reset()
    if isinstance(obs, tuple):  # gymnasium compat
        obs, _ = obs
    done = False
    truncated = False

    wealth_history = []
    action_history = []

    while not (done or truncated):
        action, mean_probs = ensemble_action(models, obs)
        action_history.append(action)

        obs, reward, done, truncated, infos = vec_env.step([action])
        info = infos[0] if isinstance(infos, list) else infos
        wealth = info.get("wealth", None)
        if wealth is not None:
            wealth_history.append(wealth)

    final_wealth = wealth_history[-1] if wealth_history else None
    print(f"Ensemble ({algo}, {version}, runs={run_ids}) final wealth: {final_wealth}")

    return {
        "final_wealth": final_wealth,
        "wealth_path": wealth_history,
        "actions": action_history,
    }

if __name__ == "__main__":
    # Usage examples:
    #   python ensemble_eval.py ppo new
    #   python ensemble_eval.py a2c new
    algo = sys.argv[1] if len(sys.argv) > 1 else "ppo"
    version = sys.argv[2] if len(sys.argv) > 2 else "new"
    run_ids = list(range(10))  # 10-member ensemble

    evaluate_ensemble(algo, version, run_ids)