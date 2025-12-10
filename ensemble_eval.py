"""
Ensemble evaluation for RL policies with per-model VecNormalize statistics.

This module evaluates an ensemble of trained policy models (PPO, A2C, TRPO)
by applying each model's saved VecNormalize observation statistics individually
before obtaining action probabilities. The ensemble performs soft voting over
per-model action probability distributions (average probabilities, then argmax).

Purpose:
- Allow evaluation where each model was trained with its own normalization
  (VecNormalize) by applying those stats at inference time without wrapping
  the environment in VecNormalize (so per-model stats are applied manually).
- Produce a trajectory of wealth and actions under the ensemble policy.

Usage examples:
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from stable_baselines3 import A2C
from sb3_contrib import TRPO, RecurrentPPO

from make_envs import make_test_env

ALGOS = {
    "ppo": RecurrentPPO,
    "a2c": A2C,
    "trpo": TRPO,
}

def make_base_env(version: str):
    """
    Return a single (unwrapped) test env instance used to attach VecNormalize stats.
    Kept simple and defaulting to the same make_test_env used elsewhere.
    """
    # do not change call sites: make_test_env has a default version="new"
    return make_test_env(version=version)

def load_vecnorm_stats(path: str) -> VecNormalize:
    """
    Load VecNormalize statistics from `path` and attach them to a tiny DummyVecEnv
    built from the project's test environment. VecNormalize.load requires a
    non-None venv with .num_envs, so we provide one here.
    """
    if path is None:
        raise ValueError("path must be a valid .pkl file for VecNormalize stats")

    # Build a single-env DummyVecEnv using the default test env (version "new")
    base_env = make_test_env()
    venv = DummyVecEnv([lambda: base_env])

    # Load VecNormalize and attach to the dummy vec-env
    vn = VecNormalize.load(path, venv)
    # Switch to evaluation mode: do not update running stats and do not normalize rewards during eval
    vn.training = False
    vn.norm_reward = False
    return vn

def normalize_obs_with_stats(obs_raw, vn: VecNormalize):
    """
    Normalize a single observation using the saved VecNormalize stats.
    Accepts obs_raw that may be:
      - numpy array shape (D,)
      - numpy array shape (1, D)
      - tuple/list like (obs, info)
    Returns an array shaped (1, D) suitable for passing to a torch policy.
    """
    # 1) extract the raw observation array if a (obs, info) tuple was passed
    if isinstance(obs_raw, (tuple, list)):
        obs_arr = obs_raw[0]
    else:
        obs_arr = obs_raw

    obs = np.asarray(obs_arr, dtype=np.float32)

    # 2) collapse leading batch dim if it's a single-environment batch
    if obs.ndim == 2 and obs.shape[0] == 1:
        obs = obs[0]

    # 3) ensure obs is 1-D (flatten if necessary)
    if obs.ndim != 1:
        obs = obs.flatten()

    # 4) pull VecNormalize stats and align shapes
    mean = np.asarray(vn.obs_rms.mean).reshape(-1)
    var = np.asarray(vn.obs_rms.var).reshape(-1)
    eps = getattr(vn.obs_rms, "eps", 1e-8)

    # If stats shape doesn't match obs, try to broadcast / flatten mean/var
    if mean.shape[0] != obs.shape[0]:
        # last-resort attempt: flatten mean/var to match obs length
        mean = mean.flatten()[: obs.shape[0]]
        var = var.flatten()[: obs.shape[0]]

    norm = (obs - mean) / np.sqrt(var + eps)
    return norm.reshape(1, -1)

def ensemble_action(models, vecnorm_stats, obs_raw, lstm_states, episode_starts):
    """
    Soft voting over models with per-model normalization.

    Accepts and returns lstm_states and episode_starts so LSTM policies
    maintain their hidden state across timesteps.
    """
    probs_list = []

    for i, (m, vn) in enumerate(zip(models, vecnorm_stats)):
        obs_norm = normalize_obs_with_stats(obs_raw, vn)     # shape (1, D)
        obs_tensor = torch.as_tensor(obs_norm, dtype=torch.float32)

        with torch.no_grad():
            # Handle RecurrentPPO (LSTM) policies specially so we preserve hidden state
            if isinstance(m, RecurrentPPO):
                # episode_start as float (1.0 or 0.0) to match sb3_contrib internals
                ep_start = torch.tensor([1.0 if bool(episode_starts[i]) else 0.0], dtype=torch.float32)

                # Initialize lstm_states[i] (tuple of torch tensors) if not set
                if lstm_states[i] is None:
                    lstm_mod = None
                    for mod in m.policy.modules():
                        if isinstance(mod, torch.nn.LSTM):
                            lstm_mod = mod
                            break
                    if lstm_mod is not None:
                        num_layers = getattr(lstm_mod, "num_layers", 1)
                        hidden_size = getattr(lstm_mod, "hidden_size", getattr(lstm_mod, "_hidden_size", 64))
                    else:
                        # conservative defaults if we couldn't find LSTM module
                        num_layers = 1
                        hidden_size = 64

                    # shape expected by sb3_contrib: (num_layers, n_seq=1, hidden_size)
                    h0 = torch.zeros(num_layers, 1, hidden_size, dtype=torch.float32)
                    c0 = torch.zeros(num_layers, 1, hidden_size, dtype=torch.float32)
                    lstm_states[i] = (h0, c0)

                try:
                    # Preferred API: get_distribution may accept lstm_states & episode_starts
                    dist_out = m.policy.get_distribution(obs_tensor, lstm_states=lstm_states[i], episode_starts=ep_start)
                    if isinstance(dist_out, tuple) and len(dist_out) == 2:
                        dist, new_states = dist_out
                        lstm_states[i] = new_states
                    else:
                        dist = dist_out
                except Exception:
                    # Last resort: use model.predict with normalized obs to preserve hidden state
                    try:
                        # model.predict expects numpy obs (no info tuple), pass obs_norm[0]
                        obs_for_predict = obs_norm.reshape(-1)
                        # episode_start as numpy float array
                        ep_np = np.array([1.0 if bool(episode_starts[i]) else 0.0], dtype=np.float32)
                        action, new_state = m.predict(obs_for_predict, state=lstm_states[i], episode_start=ep_np, deterministic=True)
                        lstm_states[i] = new_state
                        # approximate distribution as one-hot at the predicted action
                        probs = np.zeros(m.action_space.n, dtype=np.float32)
                        probs[int(np.asarray(action).reshape(-1)[0])] = 1.0
                        probs_list.append(probs)
                        episode_starts[i] = False
                        continue
                    except Exception:
                        raise

                episode_starts[i] = False

            else:
                # standard (non-recurrent) policy
                dist = m.policy.get_distribution(obs_tensor)

            # extract probs (handle both (1, K) and (K,) shapes)
            probs_np = dist.distribution.probs.detach().cpu().numpy()
            if probs_np.ndim == 2 and probs_np.shape[0] == 1:
                probs = probs_np[0]
            else:
                probs = probs_np.reshape(-1)
            probs_list.append(probs)

    mean_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    mean_probs = mean_probs / np.sum(mean_probs)
    action = int(np.argmax(mean_probs))
    return action, mean_probs, lstm_states, episode_starts

def _step_env_uniform(env, action):
    """
    Step the environment and always return (obs, rewards, dones, infos)
    where rewards and dones are length-num_envs arrays and infos is a list.

    Handles both vectorized vec_envs (has attribute 'num_envs') and raw single
    gym/gymnasium envs (returns scalars/tuples).
    """
    # Vectorized env (DummyVecEnv / VecNormalize): expects a sequence of actions
    if hasattr(env, "num_envs"):
        obs, rewards, dones, infos = env.step([action])
        return obs, rewards, dones, infos

    # Raw environment: pass scalar action and normalize outputs to vector format
    out = env.step(int(action))
    # gymnasium-style terminal/truncation: (obs, reward, terminated, truncated, info)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, np.array([reward], dtype=np.float32), np.array([done], dtype=bool), [info]
    # legacy gym: (obs, reward, done, info)
    if len(out) == 4:
        obs, reward, done, info = out
        return obs, np.array([reward], dtype=np.float32), np.array([bool(done)], dtype=bool), [info]

    raise RuntimeError("Unexpected env.step() return signature")

def evaluate_ensemble(algo: str, version: str, exp_id: int, run_ids: List[int]):
    """
    Build an ensemble from {algo}_{version}_exp{exp_id}_run{r}.zip (r in run_ids),
    each with its own VecNormalize stats, and evaluate on the test set.

    Procedure:
    1) Create a base (un-normalized) environment to get raw observations.
    2) Load each model and its corresponding VecNormalize stats.
    3) On each step, normalize the raw observation with each model's stats,
       get model action probabilities, average them (soft vote), and pick action.
    4) Step the environment with the ensemble action and record wealth/action history.
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

    # Initialize LSTM hidden states and episode_starts flags for each model
    lstm_states = [None] * len(models)
    episode_starts = np.ones((len(models),), dtype=bool)

    # Reset environment to obtain initial raw observation
    obs = vec_env.reset()  # raw obs
    done = False

    wealth_history = []
    action_history = []

    # 3) Step loop: compute ensemble action, apply to env, collect stats
    while not done:
        action, mean_probs, lstm_states, episode_starts = ensemble_action(models, vecnorm_stats, obs, lstm_states, episode_starts)
        action_history.append(action)

        obs, rewards, dones, infos = _step_env_uniform(vec_env, action)

        done = bool(dones[0])

        # If episode ended, reset LSTM hidden states for recurrent models and mark episode_starts True
        if done:
            for i, m in enumerate(models):
                if isinstance(m, RecurrentPPO):
                    lstm_states[i] = None
                episode_starts[i] = True

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

    run_ids = list(range(7))  # 7-member ensemble

    evaluate_ensemble(algo, version, exp_id, run_ids)
