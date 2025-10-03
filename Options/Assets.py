import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import math as m
import abc

# ------------------------------- 1. Spot Class --------------------------------
class Spot(object):
    def __init__(self, S, r, t, T, dt):
        S = np.asarray(S, dtype=float)
        r = np.asarray(r, dtype=float)
        T = np.asarray(T, dtype=float)
        t = np.asarray(t, dtype=float)
        dt = np.asarray(dt, dtype=float)
        
        S, r, T, t, dt = np.broadcast_arrays(S, r, T, t, dt)
        
        self.S = S
        self.r = r
        self.T = T
        self.t = t
        self.dt = dt
        
    def VarReal(self):
        """
        Compute the realised variance of the spot process. Computed by one-period squared-log returns derived from the spot price series.

        Returns
        -------
        numpy.ndarray
            Array of realised variance, same length as self.S, with NaN in initial position.
        """

        S = np.asarray(self.S, dtype=float)
        dt = self.dt

        # Convert spot prices to one-period log returns
        returns = np.diff(np.log(S)) ** 2
        returns = np.concatenate([[np.NaN],returns])/dt
        return returns
            
# ------------------------------ 2. Forward Class ------------------------------
class Forward(object):
    def __init__(self, S, r, T, t):
        self.S = S
        self.r = r
        self.T = T
        self.t = t
        
    def tau(self):
        return self.T - self.t
    
    def price(self):
        return self.S * np.exp(self.r * Forward.tau(self))

# ------------------------------ 3. Option Class -------------------------------
class Option(Spot):
    """
    Option class built on top of Spot class, inheriting features of underlying.
    
    Enables Black-Scholes pricing, Greeks computation for European options.
    
    Original Option class.

    Parameters
    ----------
    S : array_like
        Spot price.
    K : array_like
        Strike price.
    r : array_like
        Risk-free rate.
    T : array_like
        Maturity date.
    t : array_like
        Valuation date.
    sigma : array_like
        Volatility.
    style : str
        Option style: 'European'.
    type : str
        Option type: 'call'/'put'.
    dt : float, optional
        Time step (used in Spot base class).
    """
    def __init__(self, S, K, r, T, t, sigma, style, type, dt=None):
        super().__init__(S, r, t, T, dt if dt is not None else 1.0)
        
        self.K = np.asarray(K, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.style = style
        self.type = type

        self.S, self.K, self.r, self.T, self.t, self.sigma = np.broadcast_arrays(self.S, self.K, self.r, self.T, self.t, self.sigma)

    def tau(self):
        return np.maximum(self.T - self.t, 0.0)

    def d1(self):
        tau = self.tau()
        S = self.S
        K = self.K
        r = self.r
        sigma = self.sigma
        
        out = np.full_like(S, np.nan, dtype=float)
        live = tau > 0
        if np.any(live):
            out[live] = ((np.log(S[live] / K[live]) + (r[live] + 0.5 * sigma[live]**2) * tau[live]) / (sigma[live] * np.sqrt(tau[live])))
        return out

    def d2(self):
        tau = self.tau()
        out = np.full_like(self.S, np.nan, dtype=float)
        live = tau > 0
        if np.any(live):
            out[live] = self.d1()[live] - self.sigma[live] * np.sqrt(tau[live])
        return out

    def price(self):
        tau = self.tau()
        S = self.S
        K = self.K
        r = self.r
        type = self.type

        out = np.empty_like(S, dtype=float)

        expired = tau == 0
        live = ~expired

        if np.any(live):
            d1 = self.d1()[live]
            d2 = self.d2()[live]
            tau_l = tau[live]
            S_l = S[live]
            K_l = K[live]
            r_l = r[live]

            if type == 'call':
                out[live] = S_l * st.norm.cdf(d1) - K_l * np.exp(-r_l * tau_l) * st.norm.cdf(d2)
            elif type == 'put':
                out[live] = K_l * np.exp(-r_l * tau_l) * st.norm.cdf(-d2) - S_l * st.norm.cdf(-d1)
            else:
                raise(ValueError)
            
        if np.any(expired):
            if type == 'call':
                out[expired] = np.maximum(S[expired] - K[expired], 0.0)
            elif type == 'put':
                out[expired] = np.maximum(K[expired] - S[expired], 0.0)
            else:
                raise(ValueError)

        return out

    def delta(self):
        tau = self.tau()
        S = self.S
        K = self.K
        type = self.type
        
        out = np.empty_like(S, dtype=float)

        expired = tau == 0
        live = ~expired

        if np.any(expired):
            if type == 'call':
                out[expired] = (S[expired] > K[expired]).astype(float)
            elif type == 'put':
                out[expired] = -(S[expired] < K[expired]).astype(float)
            else:
                raise(ValueError)

        if np.any(live):
            d1 = self.d1()[live]
            if type == 'call':
                out[live] = st.norm.cdf(d1)
            elif type == 'put':
                out[live] = st.norm.cdf(d1) - 1.0
            else:
                raise(ValueError)

        return out

    def gamma(self):
        tau = self.tau()
        S = self.S
        sigma = self.sigma

        out = np.zeros_like(S, dtype=float)
        live = tau > 0
        if np.any(live):
            d1 = self.d1()[live]
            out[live] = st.norm.pdf(d1) / (S[live] * sigma[live] * np.sqrt(tau[live]))
        return out

    def dollar_gamma(self):
        """
        Computes dollar gamma.

        Dollar gamma is defined as 0.5 * gamma * S^2,
        where gamma is the option's gamma and S is the spot price.

        Returns
        -------
        numpy.ndarray
            Dollar gamma values, same shape as self.S.
        """
        return 0.5 * self.gamma() * (self.S ** 2)

    def theta(self):
        """
        Compute the Black-Scholes theta of the option.

        Returns
        -------
        numpy.ndarray
            Theta values, same shape as self.S.
        """
        tau = self.tau()
        S = self.S
        K = self.K
        r = self.r
        sigma = self.sigma
        type = self.type

        out = np.zeros_like(S, dtype=float)
        live = tau > 0
        if np.any(live):
            d1 = self.d1()[live]
            d2 = self.d2()[live]
            if type == 'call':
                out[live] = (-1 * (S[live] * st.norm.pdf(d1) * sigma[live]) / (2 * np.sqrt(tau[live]))
                             - r[live] * K[live] * np.exp(-r[live] * tau[live]) * st.norm.cdf(d2))
            elif type == 'put':
                out[live] = (-1 * (S[live] * st.norm.pdf(d1) * sigma[live]) / (2 * np.sqrt(tau[live]))
                             + r[live] * K[live] * np.exp(-r[live] * tau[live]) * st.norm.cdf(-d2))
            else:
                raise(ValueError)
            
        return out
        
    def vega(self):
        tau = self.tau()
        S = self.S
        sigma = self.sigma

        out = np.zeros_like(S, dtype=float)
        live = tau > 0
        if np.any(live):
            d1 = self.d1()[live]
            out[live] = st.norm.pdf(d1) * S[live] * np.sqrt(tau[live])
        return out

    def vanna(self):
        tau = self.tau()
        S = self.S
        sigma = self.sigma

        out = np.zeros_like(S, dtype=float)
        live = tau > 0
        if np.any(live):
            d1 = self.d1()[live]
            d2 = self.d2()[live]
            out[live] = st.norm.pdf(d1) * d2 / sigma[live]
        return out

    def plot(self, S_range=None):
        if S_range is None:
            S_range = np.linspace(0.01, 5, 1000)
        original_S = self.S
        self.S = np.asarray(S_range, dtype=float)
        prices = self.price()
        plt.plot(S_range, prices)
        plt.xlabel("Underlying price S")
        plt.ylabel("Option value")
        plt.show()
        self.S = original_S
        
class OptionEng(Spot):
    """Creates option class as object containing option data. Engine functionality enabled to enabled pricing in various models.

    Args:
        Spot (class): Underlying asset.
    """
    def __init__(self, S, K, r, T, t, sigma, style, type, engine, dt=None):
        super().__init__(S, r, t, T, dt if dt is not None else 1.0)
        
        self.K = np.asarray(K, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.style = style
        self.type = type
        self.engine = engine

        self.S, self.K, self.r, self.T, self.t, self.sigma = np.broadcast_arrays(self.S, self.K, self.r, self.T, self.t, self.sigma)

    def tau(self):
        return np.maximum(self.T - self.t, 0.0)
    
    def d1(self):
        return self.engine.d1(self)
    
    def d2(self):
        return self.engine.d2(self)
    
    def price(self):
        return self.engine.price(self)
    
    def delta(self):
        return self.engine.delta(self)

    def gamma(self):
        return self.engine.gamma(self)
    
    def dollarGamma(self):
        return self.engine.dollarGamma(self)
    
    def theta(self):
        return self.engine.theta(self)
        
    def vega(self):
        return self.engine.vega(self)

    def vanna(self):
        return self.engine.vanna(self)
            
# ------------------------------------ END -------------------------------------