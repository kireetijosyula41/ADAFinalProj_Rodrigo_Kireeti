import gymnasium as gym
import numpy as np

"""
gym_env_setup_new.py

Improved MDP for crypto portfolio management, informed by financial RL literature.

Key changes vs. original MDP:
- State uses a window of LOG-RETURNS instead of normalized prices.
- Reward is per-step LOG-RETURN of wealth after transaction costs.
- Optional small trade penalty discourages random churning.
- Correct terminated (bankrupt) vs truncated (end-of-data) semantics.
- Optional random_start for more varied training episodes.

State s_t:
    - window of past log-returns for all N assets: shape (window, N) -> flattened
    - one-hot current position over (N assets + cash)
    - log(current_wealth)

Action a_t:
    - discrete choice among N assets + cash:
        0..N-1 = fully allocate to that asset
        N      = hold cash

Reward r_t:
    - log(wealth_{t+1} / wealth_t) - trade_penalty * I[action != previous_position]

This aligns with common practice in RL trading: maximize cumulative log-wealth
and penalize excessive trading.
"""

class CryptoPortfolioEnvNew(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        price_array: np.ndarray,
        window_size: int = 60,
        transaction_fee: float = 0.00001,
        initial_wealth: float = 1.0,
        trade_penalty: float = 0.0,
        random_start: bool = False,
        wealth_min: float = 1e-8,
    ):
        """
        Parameters
        ----------
        price_array : np.ndarray, shape (T, N)
            Historical price data for N assets over T timesteps.
        window_size : int
            Number of past returns to include in the state.
        transaction_fee : float
            Fractional fee applied to wealth whenever the position changes.
        initial_wealth : float
            Starting wealth at the beginning of each episode.
        trade_penalty : float
            Extra penalty (in reward units) per trade to discourage churning.
        random_start : bool
            If True, start each episode at a random time index with sufficient
            history. If False, always start at t = window_size.
        wealth_min : float
            Threshold below which wealth is treated as effectively bankrupt.
        """
        super().__init__()

        assert price_array.ndim == 2, "price_array must be 2D (T, N)"
        self.prices = price_array.astype(np.float32)

        # T = number of time steps, N = number of assets
        self.T, self.N = self.prices.shape

        self.window = int(window_size)
        self.fee = float(transaction_fee)
        self.initial_wealth = float(initial_wealth)
        self.trade_penalty = float(trade_penalty)
        self.random_start = bool(random_start)
        self.wealth_min = float(wealth_min)

        # Index representing cash position
        self.cash_index = self.N

        # Precompute log-prices and log-returns
        self.log_prices = np.log(self.prices + 1e-8).astype(np.float32)
        # returns[t, j] = log(P_{t+1, j}) - log(P_{t, j}), shape (T-1, N)
        self.returns = np.diff(self.log_prices, axis=0).astype(np.float32)

        # Episode-specific state variables (set in reset)
        self.t = None            # current time index in [window, t_max]
        self.wealth = None       # current wealth
        self.position = None     # current position index (0..N for assets + cash)

        # Gym spaces
        self.action_space = gym.spaces.Discrete(self.N + 1)

        # Observation: window * N log-returns + (N+1) one-hot + 1 log-wealth
        obs_dim = self.window * self.N + (self.N + 1) + 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Need t+1 < T to compute returns; t ranges up to T-2
        self.t_max = self.T - 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Choose starting time index
        if self.random_start:
            # t in [window, t_max] inclusive
            self.t = self.np_random.integers(self.window, self.t_max + 1)
        else:
            self.t = self.window

        self.wealth = self.initial_wealth
        self.position = self.cash_index

        obs = self._get_obs()
        info = {"wealth": self.wealth, "t": self.t}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"

        # If we've already hit or passed t_max, treat as time-limit truncation
        if self.t >= self.t_max:
            obs = self._get_obs()
            info = {"wealth": self.wealth, "t": self.t}
            return obs, 0.0, False, True, info

        prev_wealth = self.wealth
        prev_position = self.position

        # 1. Compute return of the previously held asset over [t, t+1]
        if prev_position == self.cash_index:
            asset_log_ret = 0.0
        else:
            asset_log_ret = float(self.returns[self.t, prev_position])
            asset_log_ret = float(np.clip(asset_log_ret, -0.5, 0.5))

        # 2. Compute log-return directly (avoids ratio clipping)
        fee_log = 0.0
        if action != prev_position and self.fee > 0.0:
            # guard against fee == 1.0
            fee_log = float(np.log(max(1.0 - self.fee, 1e-12)))

        log_ret = asset_log_ret + fee_log

        # 3. Compute new wealth multiplicatively
        # If prev_wealth is non-positive or non-finite, treat as terminal (bankrupt)
        if not np.isfinite(prev_wealth) or prev_wealth <= 0.0:
            # immediately terminate; no meaningful return
            obs = self._get_obs()
            info = {"wealth": self.wealth, "t": self.t}
            return obs, 0.0, True, False, info

        new_wealth = prev_wealth * float(np.exp(log_ret))

        # 4. Prevent absurdly large wealth growth by terminating instead of clipping,
        #    or keep a high cap but don't use it for reward computation.
        wealth_max = 1e6
        if new_wealth <= self.wealth_min:
            # bankrupt -> terminate (no reward for this failing step)
            self.wealth = float(np.clip(new_wealth, self.wealth_min, wealth_max))
            terminated = True
            reward = 0.0
            self.position = int(action)
            self.t += 1
            obs = self._get_obs()
            info = {"wealth": self.wealth, "t": self.t}
            return obs, reward, terminated, False, info

        # Optionally clip wealth for numeric safety but do NOT use clipped ratio for reward
        self.wealth = float(np.clip(new_wealth, self.wealth_min, wealth_max))

        # 5. Per-step reward is the log-return minus trade penalty
        penalty = self.trade_penalty if action != prev_position else 0.0
        reward = float(log_ret) - penalty

        # 6. Advance state
        self.position = int(action)
        self.t += 1

        # 7. Termination / truncation
        terminated = self.wealth <= self.wealth_min
        truncated = self.t >= self.t_max

        obs = self._get_obs()
        if not np.all(np.isfinite(obs)):
            terminated = True
            reward = 0.0

        info = {"wealth": self.wealth, "t": self.t}

        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        """
        Build observation vector:

          - window of log-returns for all assets: returns[t-window : t, :]
          - one-hot for current position (N assets + cash)
          - log(current_wealth)
        """
        # returns has shape (T-1, N); at time t we have returns up to index t-1
        start = self.t - self.window
        end = self.t
        window_rets = self.returns[start:end, :]  # shape: (window, N)
        
        window_rets = np.nan_to_num(window_rets, nan=0.0, posinf=0.0, neginf=0.0)

        price_feat = window_rets.flatten().astype(np.float32)

        # one-hot position
        holding = np.zeros(self.N + 1, dtype=np.float32)
        holding[self.position] = 1.0

        # log-wealth
        w = self.wealth if np.isfinite(self.wealth) and self.wealth > 0 else self.wealth_min
        wealth_feat = np.array([np.log(w + 1e-12)], dtype=np.float32)

        obs = np.concatenate([price_feat, holding, wealth_feat])
        return obs