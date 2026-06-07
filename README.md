# Reinforcement Learning for Momentum-Driven Assets

MPCS 53112 Final Project on reinforcement learning for crypto portfolio management.

This repository implements an end-to-end workflow:
1. Collect daily OHLCV crypto data from Polygon.
2. Build an aligned multi-asset price matrix.
3. Train Gym-based portfolio agents with policy-gradient methods (PPO, A2C, TRPO).
4. Train ensembles of each method.
5. Evaluate trained policies and compare against buy-and-hold baselines.

---

## Project Structure

- `/tmp/workspace/kireetijosyula41/ADAFinalProj_Rodrigo_Kireeti/data_collection/`
  - `data_fetcher.py`: downloads OHLCV data from Polygon and saves CSVs to `data_collection/coin_data/`.
  - `coin_data/*.csv`: per-ticker historical data used for modeling.
  - `ticker_data/*`: helper files used to inspect ticker ranges.
- `/tmp/workspace/kireetijosyula41/ADAFinalProj_Rodrigo_Kireeti/build_price_array.py`
  - Converts ticker CSVs into aligned arrays:
    - `price_array_aligned.npy`
    - `price_array_dates.npy`
- Environment files:
  - `gym_env_setup_original.py`: original MDP formulation.
  - `gym_env_setup_new.py`: improved MDP formulation.
  - `make_envs.py`: train/test split and environment factory helpers.
- Training files:
  - `train_ppo.py`
  - `train_a2c.py`
  - `train_trpo.py`
- Evaluation files:
  - `evaluate_policies.py`: evaluates one trained run.
  - `ensemble_eval.py`: evaluates soft-voting ensembles.
- Ensemble runners:
  - `run_ensembling_ppo.sh`
  - `run_ensembling_a2c.sh`
  - `run_ensembling_trpo.sh`

---

## 1) Environment Setup

### Prerequisites
- Python 3.10+ recommended.
- A Polygon API key.

### Install dependencies

From `/tmp/workspace/kireetijosyula41/ADAFinalProj_Rodrigo_Kireeti`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Configure API key

Create a `.env` file in the repository root:

```bash
POLYGON_API_KEY=your_api_key_here
```

`data_collection/data_fetcher.py` loads this key via `python-dotenv`.

---

## 2) Data Collection (`data_collection`)

### What it does
`data_collection/data_fetcher.py` fetches daily OHLCV aggregates from Polygon for selected crypto tickers, standardizes columns, and saves each ticker to:

`/tmp/workspace/kireetijosyula41/ADAFinalProj_Rodrigo_Kireeti/data_collection/coin_data/<TICKER>.csv`

### Run data collection

```bash
python data_collection/data_fetcher.py
```

### Output schema
Each CSV includes fields like:
- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `vwap`
- `transactions`

---

## 3) Build the Aligned Price Array

`build_price_array.py` loads all ticker CSVs, aligns them on a shared timeline, forward-fills internal gaps, removes rows with unresolved NaNs, and saves NumPy arrays used by training/evaluation.

Run:

```bash
python build_price_array.py
```

Outputs:
- `price_array_aligned.npy` (shape `T x N`)
- `price_array_dates.npy` (aligned timestamps)

---

## 4) Gym Environment Setup

Environment construction is centralized in `make_envs.py`, which:
- Loads `price_array_aligned.npy`.
- Splits data into train/test (`70%/30%`) with overlap so the initial test window is valid.
- Returns either:
  - `CryptoPortfolioEnvOriginal` (`gym_env_setup_original.py`)
  - `CryptoPortfolioEnvNew` (`gym_env_setup_new.py`)

### Original environment (`gym_env_setup_original.py`)
- Observation: normalized windowed prices + one-hot position + wealth.
- Action: discrete full allocation to one asset or cash.
- Reward: mainly terminal wealth gain (`final_wealth - initial_wealth`).

### New environment (`gym_env_setup_new.py`)
- Observation: windowed log-return features (+ optional positional and wealth features).
- Action: same discrete full-allocation action space.
- Reward: per-step log-return with transaction-fee impact and optional trade penalty.
- Separates termination (bankrupt) vs truncation (end of data).

---

## 5) Train Policy-Gradient Models

All trainers save artifacts under:

`/tmp/workspace/kireetijosyula41/ADAFinalProj_Rodrigo_Kireeti/models/`

Each trainer:
- Wraps envs with `DummyVecEnv` + `VecNormalize`.
- Trains one algorithm.
- Saves:
  - model weights (`crypto_portfolio_<algo>_run<id>.zip`)
  - normalization stats (`vecnormalize_stats_<algo>_run<id>.pkl`)

### Train a single run

```bash
# PPO (RecurrentPPO)
python train_ppo.py new 0 0

# A2C
python train_a2c.py new 0 0

# TRPO
python train_trpo.py new 0 0
```

Arguments:
- `version`: `new` or `original`
- `run_id`: run number
- `exp_id`: experiment identifier (used for seeding)

---

## 6) Ensemble Training Workflow

Each `run_ensembling_<algo>.sh` script:
1. Sets `VERSION`, `N_RUNS` (default 7), and `EXP_ID`.
2. Trains multiple runs of that algorithm.
3. Launches ensemble evaluation after all runs finish.

Examples:

```bash
bash run_ensembling_ppo.sh
bash run_ensembling_a2c.sh
bash run_ensembling_trpo.sh
```

---

## 7) Evaluation Methods

### A) Single-run policy evaluation (`evaluate_policies.py`)

Compares one trained RL policy against:
- equal-weight buy-and-hold
- best single-asset buy-and-hold (hindsight)

Also prints action usage frequencies.

Run:

```bash
python evaluate_policies.py ppo new 0
python evaluate_policies.py a2c new 0
python evaluate_policies.py trpo new 0
```

### B) Ensemble evaluation (`ensemble_eval.py`)

Builds an ensemble of multiple runs for one algorithm:
- Loads each model and its own `VecNormalize` stats.
- Normalizes observations per-model.
- Applies soft voting on action probabilities.
- Executes the chosen action in the environment and reports wealth trajectory/final wealth.

Run:

```bash
python ensemble_eval.py ppo new 0
python ensemble_eval.py a2c new 0
python ensemble_eval.py trpo new 0
```

By default, it evaluates a 7-member ensemble (`run_ids = 0..6`).

---

## Typical End-to-End Pipeline

From repository root:

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Collect raw data
python data_collection/data_fetcher.py

# 3) Build aligned training array
python build_price_array.py

# 4) Train ensembles and evaluate
bash run_ensembling_ppo.sh
bash run_ensembling_a2c.sh
bash run_ensembling_trpo.sh

# 5) Optional single-run evaluation
python evaluate_policies.py ppo new 0
```

---

## Notes

- Ensure `POLYGON_API_KEY` is set before running data collection.
- Training is compute-intensive; adjust timesteps and run counts as needed.
- The default configuration uses the `new` environment version in ensemble scripts.
