"""
make_envs.py

Helpers to construct training and testing Gym environments for the
crypto portfolio experiments.

Purpose:
- Load the aligned price array used across experiments.
- Split the price array into a training slice and a test slice (with overlap
  so the observation window is valid at episode start).
- Provide factory functions that return configured environment instances
  (either the "original" MDP or the "new" improved MDP).

This module is intentionally lightweight and keeps the call signatures used
by training/evaluation scripts unchanged.
"""

import numpy as np
from gym_env_setup_original import CryptoPortfolioEnvOriginal
from gym_env_setup_new import CryptoPortfolioEnvNew

# Default hyperparameters used when constructing envs via the factory functions.
WINDOW_SIZE = 60
FEE = 0.000001

def _load_price_array():
    """
    Load the master price array used for experiments.

    Returns
    -------
    np.ndarray
        A float32 array loaded from PRICE_ARRAY_PATH (expects
        "price_array_aligned.npy" in the repo root).
    """
    return np.load("price_array_aligned.npy").astype(np.float32)

def _split_train_test(price_array, train_fraction=0.7):
    """
    Split the provided price array into train and test portions.

    Notes:
    - The test_prices slice includes WINDOW_SIZE rows of overlap at the start
      so that callers can request an initial observation window without
      running off the start of the array.

    Parameters
    ----------
    price_array : np.ndarray
        Full time-series price array (T x N).
    train_fraction : float
        Fraction of the data to allocate to training.

    Returns
    -------
    (train_prices, test_prices) : tuple of np.ndarray
        Slices of price_array for training and testing.
    """
    T = price_array.shape[0]
    train_T = int(train_fraction * T)
    train_prices = price_array[:train_T]
    test_prices = price_array[train_T - WINDOW_SIZE:]
    return train_prices, test_prices

def _env_classes(version):
    """
    Map the textual version identifier to the environment class.

    Accepts "original" or "new" and returns the corresponding class object.
    Raises ValueError for unknown versions.
    """
    match version:
        case "original":
            return CryptoPortfolioEnvOriginal
        case "new":
            return CryptoPortfolioEnvNew
        case _:
            raise ValueError(f"Unknown version: {version}")

def make_train_env(version="new", include_wealth: bool = True):
    """
    Construct a training environment instance.

    This function:
    - loads the stored price array,
    - extracts the training slice,
    - instantiates the requested env class with default parameters.

    Parameters
    ----------
    version : str
        "new" or "original" to select the env implementation.
    include_wealth : bool
        Passed through to the env constructor to control whether wealth is
        included in the observation.

    Returns
    -------
    gym.Env
        A configured environment instance ready for training.
    """
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
    """
    Construct a testing environment instance.

    Mirrors make_train_env but uses the held-out test slice. The test slice
    includes an overlap so the env can provide a valid initial observation.
    """
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
    # Quick local smoke test: prints the observation shape for the chosen version.
    print("Usage: python make_envs.py [original|new]")
    import sys
    env = make_train_env(sys.argv[1])
    obs, info = env.reset()
    print("Mode", sys.argv[1], "Obs shape:", obs.shape, "| Info:", info)