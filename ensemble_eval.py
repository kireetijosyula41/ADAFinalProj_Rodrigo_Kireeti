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


def make_eval_vec_env_for_stats(version: str, algo: str, stats_run: int):
    """
    Build a test VecEnv and load VecNormalize stats from a specific run.

    This env will use vecnormalize_stats_{algo}_run{stats_run}.pkl
    to normalize observations and (optionally) rewards.
    """
    def _init():
        return make_test_env(version)

    base_env = DummyVecEnv([_init])

    stats_path = f"./models/vecnormalize_stats_{algo}_run{stats_run}.pkl"
    print(f"  Loading VecNormalize stats from: {stats_path}")
    vec_env = VecNormalize.load(stats_path, base_env)

    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def ensemble_action(models, obs):
    """
    Soft voting: average action probabilities over models.

    obs: np.ndarray of shape (1, obs_dim) as returned by VecEnv.
    """
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
    probs_list = []

    for m in models:
        with torch.no_grad():
            dist = m.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy()[0]
            probs_list.append(probs)

    mean_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    mean_probs = mean_probs / np.sum(mean_probs)  # safety renormalization
    action = int(np.argmax(mean_probs))
    return action, mean_probs


def evaluate_ensemble_with_stats_run(
    algo: str,
    version: str,
    run_ids: List[int],
    stats_run: int,
):
    """
    Evaluate ensemble using one particular VecNormalize stats file.

    - algo: "ppo" / "a2c" / "trpo"
    - version: "new" / "original"
    - run_ids: model run indices, e.g. [0..9]
    - stats_run: which run's VecNormalize stats to use for the env
    """
    ModelClass = ALGOS[algo]
    print(f"\n=== Evaluating ensemble with stats from run {stats_run} ===")

    # 1) Build eval env using stats from this stats_run
    vec_env = make_eval_vec_env_for_stats(version, algo, stats_run)

    # 2) Load all models for this algo+version, attached to this env
    models = []
    for r in run_ids:
        model_path = f"./models/crypto_portfolio_{algo}_run{r}"
        print(f"  Loading model: {model_path}")
        model = ModelClass.load(model_path, env=vec_env)
        models.append(model)

    # 3) Run a single rollout using soft-voting ensemble
    obs = vec_env.reset()  # VecEnv API: obs only
    done = False

    wealth_history = []
    action_history = []

    while not done:
        action, mean_probs = ensemble_action(models, obs)
        action_history.append(action)

        obs, rewards, dones, infos = vec_env.step([action])
        done = bool(dones[0])

        info = infos[0]
        wealth = info.get("wealth", None)
        if wealth is not None:
            wealth_history.append(float(wealth))

    final_wealth = wealth_history[-1] if wealth_history else None
    print(f"  Final wealth (stats_run={stats_run}): {final_wealth}")
    print(f"  Total steps: {len(action_history)}")

    return final_wealth, wealth_history, action_history


def evaluate_ensemble_all_stats(algo: str, version: str, run_ids: List[int]):
    """
    For each stats_run in run_ids, evaluate the same model ensemble but
    with that run's VecNormalize stats.

    Returns a dict mapping stats_run -> final_wealth.
    """
    results = {}
    for stats_run in run_ids:
        final_w, _, _ = evaluate_ensemble_with_stats_run(
            algo=algo,
            version=version,
            run_ids=run_ids,
            stats_run=stats_run,
        )
        results[stats_run] = final_w

    print("\nSummary of final wealth for each stats_run:")
    for k in sorted(results.keys()):
        print(f"  stats_run {k}: {results[k]}")
    return results


if __name__ == "__main__":
    algo = sys.argv[1] if len(sys.argv) > 1 else "ppo"
    version = sys.argv[2] if len(sys.argv) > 2 else "new"

    # we assume 10 runs: 0..9
    run_ids = list(range(10))

    evaluate_ensemble_all_stats(algo, version, run_ids)