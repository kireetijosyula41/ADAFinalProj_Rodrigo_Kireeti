# make_envs.py
import numpy as np
from gym_env_setup_original import CryptoPortfolioEnvOriginal
from gym_env_setup_new import CryptoPortfolioEnvNew

# make_envs.py (baseline / original MDP wiring)

WINDOW_SIZE = 180
FEE = 0.000001

def make_train_env():
    """
    Create a single training environment using the original MDP.

    Uses ~70% of the available history for training.
    """
    price_array = np.load("price_array_aligned.npy").astype(np.float32)
    T = price_array.shape[0]
    # Instead of hardcoding 1000 days, use 70% of data for training
    train_T = int(0.7 * T)

    train_prices = price_array[:train_T]

    env = CryptoPortfolioEnvOriginal(
        price_array=train_prices,
        window_size=WINDOW_SIZE,
        transaction_fee=FEE,
        initial_wealth=1.0,
    )
    return env

def make_test_env():
    """
    Create a single test environment using the original MDP.

    The test segment starts after the training data, but includes an overlap
    of WINDOW_SIZE days so the observation window is well-defined.
    """
    price_array = np.load("price_array_aligned.npy").astype(np.float32)
    T = price_array.shape[0]
    train_T = int(0.7 * T)

    # include overlap for the history window
    start_idx = max(train_T - WINDOW_SIZE, 0)
    test_prices = price_array[start_idx:]

    env = CryptoPortfolioEnvOriginal(
        price_array=test_prices,
        window_size=WINDOW_SIZE,
        transaction_fee=FEE,
        initial_wealth=1.0,
    )
    return env

def make_train_env_new():
    """
    Create a single training environment using the original MDP.

    Uses ~70% of the available history for training.
    """
    price_array = np.load("price_array_aligned.npy").astype(np.float32)
    T = price_array.shape[0]
    # Instead of hardcoding 1000 days, use 70% of data for training
    train_T = int(0.8 * T)

    train_prices = price_array[:train_T]

    env = CryptoPortfolioEnvNew(
        price_array=train_prices,
        window_size=WINDOW_SIZE,
        transaction_fee=FEE,
        initial_wealth=1.0,
    )
    return env

def make_test_env_new():
    """
    Create a single test environment using the original MDP.

    The test segment starts after the training data, but includes an overlap
    of WINDOW_SIZE days so the observation window is well-defined.
    """
    price_array = np.load("price_array_aligned.npy").astype(np.float32)
    T = price_array.shape[0]
    train_T = int(0.8 * T)

    # include overlap for the history window
    start_idx = max(train_T - WINDOW_SIZE, 0)
    test_prices = price_array[start_idx:]

    env = CryptoPortfolioEnvNew(
        price_array=test_prices,
        window_size=WINDOW_SIZE,
        transaction_fee=FEE,
        initial_wealth=1.0,
    )
    return env

if __name__ == "__main__":
    print("Usage: python make_envs.py [original|new]")
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "new":
        env = make_train_env_new()
    else:
        env = make_train_env()
    obs, info = env.reset()
    print("Mode", sys.argv[1], "Obs shape:", obs.shape, "| Info:", info)