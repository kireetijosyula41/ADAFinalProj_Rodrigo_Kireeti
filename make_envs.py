# make_envs.py
import numpy as np
from gym_env_setup_original import CryptoPortfolioEnvOriginal
from gym_env_setup_new import CryptoPortfolioEnvNew

# make_envs.py (baseline / original MDP wiring)

WINDOW_SIZE = 60
FEE = 0.000001

def _load_price_array():
    return np.load("price_array_aligned.npy").astype(np.float32)

def _split_train_test(price_array, train_fraction=0.7):
    T = price_array.shape[0]
    train_T = int(train_fraction * T)
    train_prices = price_array[:train_T]
    test_prices = price_array[train_T - WINDOW_SIZE:]
    return train_prices, test_prices

def _env_classes(version):
    match version:
        case "original":
            return CryptoPortfolioEnvOriginal
        case "new":
            return CryptoPortfolioEnvNew
        case _:
            raise ValueError(f"Unknown version: {version}")

def make_train_env(version="new", include_wealth: bool = True):
    price_array = _load_price_array()
    train_prices, _ = _split_train_test(price_array, train_fraction=0.7)
    EnvClass = _env_classes(version)
    env = EnvClass(
        price_array=train_prices,
        window_size=WINDOW_SIZE,
        transaction_fee=FEE,
        initial_wealth=1.0,
        include_wealth=include_wealth,   # pass through
        include_positional=True,      # NEW
    )
    return env

def make_test_env(version="new", include_wealth: bool = True):
    price_array = _load_price_array()
    _, test_prices = _split_train_test(price_array, train_fraction=0.7)
    EnvClass = _env_classes(version)
    env = EnvClass(
        price_array=test_prices,
        window_size=WINDOW_SIZE,
        transaction_fee=FEE,
        initial_wealth=1.0,
        include_wealth=include_wealth,   # pass through
        include_positional=True,      # NEW
    )
    return env

if __name__ == "__main__":
    print("Usage: python make_envs.py [original|new]")
    import sys
    env = make_train_env(sys.argv[1])
    obs, info = env.reset()
    print("Mode", sys.argv[1], "Obs shape:", obs.shape, "| Info:", info)