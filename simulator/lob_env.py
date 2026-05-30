import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LOBEnv(gym.Env):
    """
    Synthetic Limit Order Book Environment for High-Frequency Market Making.
    Simulates mid-price as Brownian motion and order arrivals via Poisson process.
    """
    def __init__(self, sigma=0.1, dt=1.0, k=1.5, A=140, gamma=0.1, max_steps=200):
        super(LOBEnv, self).__init__()
        
        # AS parameters
        self.sigma = sigma  # Volatility of the asset
        self.dt = dt        # Time increment
        self.k = k          # Order execution probability decay factor
        self.A = A          # Order arrival intensity
        self.gamma = gamma  # Risk aversion parameter
        
        self.max_steps = max_steps
        self.current_step = 0
        
        self.S0 = 100.0     # Initial mid-price
        self.S = self.S0
        self.q = 0          # Inventory position
        self.cash = 0.0     # Cash account
        
        # Action space: [ask_spread, bid_spread]
        # Spread is the distance from the mid-price.
        self.action_space = spaces.Box(low=0.0, high=5.0, shape=(2,), dtype=np.float32)
        
        # State space: [Mid-price, Inventory (q), Time remaining (T-t), OFI, CumNotional]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.S = self.S0
        self.q = 0
        self.cash = 0.0
        self.current_step = 0
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        time_rem = (self.max_steps - self.current_step) * self.dt
        # OFI and CumNotional are simulated randomly for this synthetic environment to provide
        # dense state features for the A2C network.
        ofi = np.random.normal(0, 1)
        cum_notional = np.random.uniform(10, 100)
        return np.array([self.S, float(self.q), time_rem, ofi, cum_notional], dtype=np.float32)
        
    def step(self, action):
        ask_spread, bid_spread = action
        
        # Ensure spreads are non-negative
        ask_spread = max(0.01, ask_spread)
        bid_spread = max(0.01, bid_spread)
        
        # Quote prices
        p_a = self.S + ask_spread
        p_b = self.S - bid_spread
        
        # Simulate intensities based on AS Poisson process
        lambda_a = self.A * np.exp(-self.k * ask_spread)
        lambda_b = self.A * np.exp(-self.k * bid_spread)
        
        # Simulate fill probabilities in this dt
        prob_a = 1.0 - np.exp(-lambda_a * self.dt)
        prob_b = 1.0 - np.exp(-lambda_b * self.dt)
        
        fill_a = np.random.rand() < prob_a
        fill_b = np.random.rand() < prob_b
        
        # Update inventory and cash
        step_cash_reward = 0.0
        if fill_a:
            self.q -= 1
            self.cash += p_a
            step_cash_reward += p_a - self.S # Spread capture relative to mid
        if fill_b:
            self.q += 1
            self.cash -= p_b
            step_cash_reward += self.S - p_b # Spread capture relative to mid
            
        # Update mid-price (Brownian motion)
        dW = np.random.normal(0, np.sqrt(self.dt))
        self.S += self.sigma * dW
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
            
        # Reward function balancing PnL, Inventory Penalty, and Quoting Penalty
        inventory_penalty = self.gamma * (self.q ** 2)
        spread_penalty = 0.01 * (ask_spread + bid_spread)
        reward = step_cash_reward - inventory_penalty - spread_penalty
        
        # Calculate true mark-to-market PnL
        pnl = self.cash + (self.q * self.S)
        
        # Terminal adjustment
        if done:
            reward += pnl * 0.01 # minor reward for final wealth
            
        info = {
            'cash': self.cash,
            'inventory': self.q,
            'mid_price': self.S,
            'pnl': pnl
        }
        
        return self._get_obs(), reward, done, False, info
