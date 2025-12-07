import gymnasium as gym
import numpy as np

"""
Original MDP environment as defined in the midterm presentation/proposal.

- State:
    - Window of past PRICES for all N assets (normalized to the first price in window)
    - Current holding as one-hot over (N assets + cash)
    - Current WEALTH (raw, not log)
- Action:
    - Discrete choice among N assets + cash
    - 0..N-1: hold that asset fully
    - N: hold cash
- Transaction fee:
    * If action != previous position, apply multiplicative fee:
          wealth <- wealth * (1 - fee)
- Reward:
    * 0 at each intermediate step
    * At terminal step (bankrupt or end of data):
          reward = final_wealth - initial_wealth
"""

class CryptoPortfolioEnvOriginal(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, 
                 price_array: np.ndarray, 
                 window_size: int = 60, 
                 transaction_fee: float = 0.0001, 
                 initial_wealth: float = 1.0):
        # Inherit from gym.Env
        super().__init__()
        self.prices = price_array.astype(np.float32)
        self.T, self.N = self.prices.shape

        self.window = window_size
        self.fee = float(transaction_fee)
        self.initial_wealth = float(initial_wealth)

        self.cash_index = self.N

        # self.t = step number
        # self
        # Will be set in the reset() function
        self.t = None
        self.wealth = None
        self.position = None

        # Action space: discrete choices among N assets + cash
        self.action_space = gym.spaces.Discrete(self.N + 1)
        
        # Observation:
        #   window * N price features
        #   + (N + 1) one-hot position
        #   + 1 scalar wealth
        obs_dim = self.window * self.N + (self.N + 1) + 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # last usable time index (we need t+1 < T to compute a return)
        self.t_max = self.T - 2

    def reset(self, seed=None, options=None):

        # Reset environment state
        super().reset(seed=seed)

        # start at t = window
        self.t = self.window
        self.wealth = self.initial_wealth
        self.position = self.cash_index

        obs = self._get_obs()
        info = {"wealth": self.wealth, "t": self.t}
        return obs, info
    
    def _get_obs(self):
        # 1. window of past prices: [t-window, ..., t-1]
        start = self.t - self.window
        end = self.t
        window_prices = self.prices[start:end, :]
        base = window_prices[0:1, :]
        rel_window = (window_prices / (base + 1e-8)) - 1.0
        # flatten to length window * N
        price_feat = rel_window.flatten().astype(np.float32)

        # one-hot for current position (0..N, N=cash)
        holding = np.zeros(self.N + 1, dtype=np.float32)
        holding[self.position] = 1.0

        # 3. log(wealth)
        wealth_feat = np.array([self.wealth], dtype=np.float32)

        obs = np.concatenate([price_feat, holding, wealth_feat])
        return obs
    
    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"
        # If we are at t_max already, we can't go further
        # but Gymnasium expects step() to be callable; so we
        # treat this as terminal with zero reward.
        if self.t >= self.t_max:
            obs = self._get_obs()
            info = {"wealth": self.wealth, "t": self.t}
            return obs, 0.0, False, True, info
        
        prev_wealth = self.wealth
        prev_position = self.position

        # Compute asset return from the position chosen at time t (action)
        # i.e. the allocation is effective for period [t, t+1]
        if action == self.cash_index:
            asset_ret = 0.0
        else:
            p_t = self.prices[self.t, action]
            p_tp1 = self.prices[self.t + 1, action]
            asset_ret = (p_tp1 / (p_t + 1e-12)) - 1.0

        # Wealth update before fee
        new_wealth = prev_wealth * (1.0 + asset_ret)

        # Apply transaction fee multiplicatively if we change position (once)
        if action != prev_position:
            new_wealth *= (1.0 - self.fee)

        # Avoid negative or zero wealth blowing up logs
        new_wealth = max(new_wealth, 1e-12)

        # 4. Advance internal state
        self.wealth = new_wealth
        self.position = int(action)
        self.t += 1

        # Termination conditions:
        terminated = self.wealth <= 1e-12
        truncated = self.t >= self.t_max

        if terminated or truncated:
            reward = float(self.wealth - self.initial_wealth)
        else:
            reward = 0.0

        obs = self._get_obs()
        info = {"wealth": self.wealth, "t": self.t}
        return obs, reward, terminated, truncated, info

