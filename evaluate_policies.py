# evaluate_policies.py
import numpy as np
from stable_baselines3 import PPO, A2C
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from make_envs import make_test_env_new

from make_envs import make_test_env

# Purpose of this file: evaluate trained RL policies against buy-and-hold baselines
# Used by running `python evaluate_policies.py` after training with train_ppo.py

TICKERS = ["EOSUSD", "ZRXUSD", "NEOUSD", "TRXUSD", "OMGUSD", "BATUSD", "ADAUSD", "QTUMUSD", "XTZUSD", "DOGEUSD"]

# TICKERS = [
#     "GALAUSD", "DOGEUSD", "MANAUSD", "SHIBUSD", "AVAXUSD",
#     "TRXUSD", "CVCUSD", "SOLUSD", "CHZUSD", "ENJUSD",
# ]

PRICE_ARRAY_PATH = "price_array_aligned.npy"

WINDOW_SIZE = 60

def make_eval_env(version, VECNORM_PATH):
    """Create a VecNormalize-wrapped test environment for evaluation.

    We recreate a DummyVecEnv around make_test_env, then load the
    VecNormalize statistics saved during training so that observations
    are normalized in the same way as during training.
    """
    if version == "new":
        def _make_env():
            return make_test_env_new()
    else:
        def _make_env():
            return make_test_env()

    vec_env = DummyVecEnv([_make_env])
    vec_env = VecNormalize.load(VECNORM_PATH, vec_env)
    # Set to evaluation mode: do not update running stats, and do not normalize rewards
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env

def run_trained_agent(vec_env, model):
    """Run the trained PPO agent once on the test environment.

    Returns
    -------
    wealth_path : np.ndarray, shape (T_steps,)
        Wealth trajectory over the test episode (starting from 1.0).
    actions_taken : np.ndarray, shape (T_steps,)
        Discrete actions chosen by the policy at each step.
    """
    obs = vec_env.reset()
    wealth_path = [1.0]  # we know initial_wealth is 1.0 in the env
    actions_taken = []
    rewards = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)

        done = bool(dones[0])
        info = infos[0]
        wealth = float(info.get("wealth", np.nan))
        wealth_path.append(wealth)
        rewards.append(float(reward[0]))
        actions_taken.append(int(action[0]))

    final_list = []
    for i in zip(wealth_path, actions_taken, rewards):
        final_list.append(i)
    return np.array(wealth_path, dtype=np.float32), np.array(actions_taken, dtype=np.int32)

def buy_and_hold_baselines():
    """Compute buy-and-hold baselines on the same test segment.

    For each coin j, we simulate investing all wealth in that coin at the
    first tradable day of the test period and holding until the end.
    We also compute an equal-weight portfolio that holds all coins equally.

    Returns
    -------
    wealth_each : np.ndarray, shape (T_test, N_assets)
        Wealth paths for buy-and-hold of each individual coin.
    equal_weight : np.ndarray, shape (T_test,)
        Wealth path for an equal-weight buy-and-hold portfolio.
    """
    price_array = np.load(PRICE_ARRAY_PATH).astype(np.float32)
    T, N = price_array.shape
    assert N == len(TICKERS), "Ticker list length must match number of columns in price_array."

    # Use same train/test split as make_envs.make_test_env
    train_T = int(0.7 * T)
    start_idx = max(train_T - WINDOW_SIZE, 0)
    test_prices = price_array[start_idx:]  # shape: (T_test_full, N)

    # The environment starts trading after it has seen WINDOW_SIZE days,
    # i.e., at local index WINDOW_SIZE within test_prices.
    tradable_prices = test_prices[WINDOW_SIZE:]  # shape: (T_tradable, N)
    T_tradable = tradable_prices.shape[0]

    wealth_each = np.zeros((T_tradable, N), dtype=np.float32)

    for j in range(N):
        p0 = tradable_prices[0, j]
        # Avoid division by zero
        wealth_each[:, j] = tradable_prices[:, j] / (p0 + 1e-8)

    equal_weight = wealth_each.mean(axis=1)
    return wealth_each, equal_weight

def print_action_summary(actions_taken):
    """Print a simple summary of how often each action was taken."""
    unique, counts = np.unique(actions_taken, return_counts=True)
    total = actions_taken.shape[0]
    print("\nAction usage (count and percentage):")
    for a, c in zip(unique, counts):
        if a == len(TICKERS):  # cash index
            name = "CASH"
        else:
            name = TICKERS[a]
        pct = 100.0 * c / total
        print(f"  Action {a:2d} ({name:7s}): {c:4d} steps ({pct:5.1f}%)")


def main(MODEL_PATH, VECNORM_PATH, model_test, version):
    print("Usage: python evaluate_policies.py [a2c|trpo|ppo] [original|new]")
    # 1. Build evaluation environment and load model
    vec_env = make_eval_env(version, VECNORM_PATH)
    match model_test:
        case "ppo":
            model = PPO.load(MODEL_PATH, env=vec_env)
        case "a2c":
            model = A2C.load(MODEL_PATH, env=vec_env)
        case "trpo":
            model = TRPO.load(MODEL_PATH, env=vec_env)
        case _:
            raise ValueError(f"Unknown model type: {version}")

    # 2. Run trained agent on the test env
    rl_path, actions_taken = run_trained_agent(vec_env, model)

    # 3. Compute buy-and-hold baselines
    wealth_each, equal_weight = buy_and_hold_baselines()

    final_rl = float(rl_path[-1])
    final_eq = float(equal_weight[-1])

    # Best single-coin (in hindsight)
    final_each = wealth_each[-1]  # final wealth for each coin
    best_idx = int(final_each.argmax())
    final_best_coin = float(final_each[best_idx])

    print("\n=== Final Wealth Comparison (Original MDP) ===")
    print(f"RL policy final wealth           : {final_rl:.4f}")
    print(f"Equal-weight buy&hold final wealth: {final_eq:.4f}")
    print(f"Best single coin (hindsight)    : {TICKERS[best_idx]} with final wealth {final_best_coin:.4f}")

    # 4. Optional: summarize which actions the agent used
    print_action_summary(actions_taken)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        print(sys.argv)
        model_test = sys.argv[1]
        version = sys.argv[2]
    MODEL_PATH = f"crypto_portfolio_{model_test}.zip"
    VECNORM_PATH = f"vecnormalize_stats_{model_test}.pkl"
    main(MODEL_PATH, VECNORM_PATH, model_test, version)