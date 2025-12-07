"""
Ensemble evaluation with per-model VecNormalize stats (correct logic).

Usage:
    python ensemble_eval.py ppo new 0
    python ensemble_eval.py a2c new 0
    python ensemble_eval.py trpo new 0

where:
    algo    = "ppo" | "a2c" | "trpo"
    version = "new" | "original"
    exp_id  = experiment index (0, 1, ...)
"""
import sys
from typing import List

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

def make_base_env(version: str):
    """
    Plain test VecEnv (NO VecNormalize) so we can apply per-model normalization.
    """
    def _init():
        return make_test_env(version)

    vec_env = DummyVecEnv([_init])
    return vec_env

def load_vecnorm_stats(path: str) -> VecNormalize:
    """
    Load VecNormalize only to recover its statistics; env is not used.
    """
    vn = VecNormalize.load(path, venv=None)
    return vn

def normalize_obs_with_stats(obs: np.ndarray, vn: VecNormalize) -> np.ndarray:
    """
    Apply VecNormalize's observation normalization manually:

        obs_norm = clip( (obs - mean) / sqrt(var + eps), [-clip_obs, clip_obs] )

    obs: shape (1, obs_dim)
    """
    mean = vn.obs_rms.mean
    var = vn.obs_rms.var
    eps = vn.epsilon
    clip = vn.clip_obs

    norm = (obs - mean) / np.sqrt(var + eps)
    norm = np.clip(norm, -clip, clip)
    return norm

def ensemble_action(models, vecnorm_stats, obs_raw):
    """
    Soft voting over models with per-model normalization.

    models        : list of SB3 models
    vecnorm_stats : list of VecNormalize objects (stats only)
    obs_raw       : np.ndarray (1, obs_dim) from the base env
    """
    probs_list = []

    for m, vn in zip(models, vecnorm_stats):
        obs_norm = normalize_obs_with_stats(obs_raw, vn)
        obs_tensor = torch.as_tensor(obs_norm, dtype=torch.float32)

        with torch.no_grad():
            dist = m.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy()[0]
            probs_list.append(probs)

    mean_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    mean_probs = mean_probs / np.sum(mean_probs)  # safety renorm
    action = int(np.argmax(mean_probs))
    return action, mean_probs

def evaluate_ensemble(algo: str, version: str, exp_id: int, run_ids: List[int]):
    """
    Build an ensemble from {algo}_{version}_exp{exp_id}_run{r}.zip (r in run_ids),
    each with its own VecNormalize stats, and evaluate on the test set.
    """
    ModelClass = ALGOS[algo]

    print(f"=== Evaluating {algo.upper()} ensemble: version={version}, exp_id={exp_id} ===")
    print(f"Runs: {run_ids}")

    # 1) Base env without VecNormalize
    vec_env = make_base_env(version)

    # 2) Load all models + their stats
    models = []
    vecnorm_stats = []
    for r in run_ids:
        model_path = f"./models/crypto_portfolio_{algo}_run{r}.zip"
        stats_path = f"./models/vecnormalize_stats_{algo}_run{r}.pkl"

        print(f"  Loading model: {model_path}")
        model = ModelClass.load(model_path, env=vec_env)
        models.append(model)

        print(f"  Loading VecNormalize stats: {stats_path}")
        vn = load_vecnorm_stats(stats_path)
        vecnorm_stats.append(vn)

    obs = vec_env.reset()  # raw obs
    done = False

    wealth_history = []
    action_history = []

    while not done:
        action, mean_probs = ensemble_action(models, vecnorm_stats, obs)
        action_history.append(action)

        obs, rewards, dones, infos = vec_env.step([action])
        done = bool(dones[0])

        info = infos[0]
        wealth = info.get("wealth", None)
        if wealth is not None:
            wealth_history.append(float(wealth))

    final_wealth = wealth_history[-1] if wealth_history else None
    print(f"\nFinal wealth (algo={algo}, version={version}, exp_id={exp_id}): {final_wealth}")
    print(f"Total steps: {len(action_history)}")

    return {
        "final_wealth": final_wealth,
        "wealth_path": wealth_history,
        "actions": action_history,
    }

if __name__ == "__main__":
    # Defaults: algo=ppo, version=new, exp_id=0
    algo = sys.argv[1] if len(sys.argv) > 1 else "ppo"
    version = sys.argv[2] if len(sys.argv) > 2 else "new"
    exp_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    run_ids = list(range(10))  # 10-member ensemble

    evaluate_ensemble(algo, version, exp_id, run_ids)