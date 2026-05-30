import numpy as np

class AvellanedaStoikovModel:
    """
    Classical Avellaneda-Stoikov (AS) Stochastic Control Model for High-Frequency Market Making.
    Provides mathematically derived closed-form approximations for optimal quoting.
    """
    def __init__(self, gamma=0.1, sigma=0.1, k=1.5, dt=1.0):
        self.gamma = gamma   # Risk aversion parameter
        self.sigma = sigma   # Volatility of the asset
        self.k = k           # Order book liquidity parameter (intensity decay)
        self.dt = dt         # Time step size
        
    def get_quotes(self, S, q, time_remaining):
        """
        Calculate optimal ask and bid spreads from the mid-price based on AS equations.
        
        Args:
            S: Current market mid-price
            q: Current inventory position
            time_remaining: (T - t) Time until the end of the trading horizon
            
        Returns:
            ask_spread, bid_spread (distance from mid-price)
        """
        # Reservation price: r(s, t) = s - q * gamma * sigma^2 * (T - t)
        res_price = S - (q * self.gamma * (self.sigma ** 2) * time_remaining)
        
        # Optimal spread: spread = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + (gamma/k))
        optimal_spread = self.gamma * (self.sigma ** 2) * time_remaining + (2 / self.gamma) * np.log(1 + (self.gamma / self.k))
        
        # Quotes are placed symmetrically around the reservation price
        half_spread = optimal_spread / 2.0
        
        ask_price = res_price + half_spread
        bid_price = res_price - half_spread
        
        # Convert absolute prices back into spreads from mid-price (for action space compatibility)
        ask_spread = ask_price - S
        bid_spread = S - bid_price
        
        # Safety constraint: In real life, spread cannot be negative
        ask_spread = max(0.01, ask_spread)
        bid_spread = max(0.01, bid_spread)
        
        return ask_spread, bid_spread
