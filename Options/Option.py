import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import math as m
import abc

# ------------------------------- 1. Spot Class --------------------------------
class Spot(object):
    def __init__(self, S, r, t, T, deltaT):
        S = np.asarray(S, dtype=float)
        r = np.asarray(r, dtype=float)
        T = np.asarray(T, dtype=float)
        t = np.asarray(t, dtype=float)
        deltaT = np.asarray(deltaT, dtype=float)
        
        S, r, T, t, deltaT = np.broadcast_arrays(S, r, T, t, deltaT)
        
        self.S = S
        self.r = r
        self.T = T
        self.t = t
        self.deltaT = deltaT
        
    def VarReal(self):
        """
        Compute the realised variance of the spot process. Computed by one-period squared-log returns derived from the spot price series.

        Returns
        -------
        numpy.ndarray
            Array of realised variance, same length as self.S, with NaN in initial position.
        """

        S = np.asarray(self.S, dtype=float)
        deltaT = self.deltaT

        # Convert spot prices to one-period log returns
        returns = np.diff(np.log(S)) ** 2
        returns = np.concatenate([[np.NaN],returns])/deltaT
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
    deltaT : float, optional
        Time step (used in Spot base class).
    """
    def __init__(self, S, K, r, T, t, sigma, style, type, deltaT=None):
        super().__init__(S, r, t, T, deltaT if deltaT is not None else 1.0)
        
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
    def __init__(self, S, K, r, T, t, sigma, style, type, engine, deltaT=None):
        super().__init__(S, r, t, T, deltaT if deltaT is not None else 1.0)
        
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
        
# -------------------------- 4. Pricing Engine Class ---------------------------
class PricingEngine(abc.ABC):
    """
    Abstract base class for option pricing engines.

    This class defines the interface for pricing models applied to options.
    
    Concrete subclasses implement all abstract methods to provide consistent pricing and Greek calculations across different modelling environments.

    Methods
    -------
    price(option) : np.ndarray
        Compute the fair value of the option.
    delta(option) : np.ndarray
        Compute the option delta (∂V/∂S).
    gamma(option) : np.ndarray
        Compute the option gamma (∂²V/∂S²).
    dollarGamma(option) : np.ndarray
        Compute the option dollar gamma = 0.5 * gamma * S².
    theta(option) : np.ndarray
        Compute the option theta (∂V/∂t, time decay).
    vega(option) : np.ndarray
        Compute the option vega (∂V/∂σ).
    vanna(option) : np.ndarray
        Compute the option vanna (∂²V/∂S∂σ).

    Notes
    -----
    - This class should not be instantiated directly.
    - Subclasses must provide model-specific implementations.
    - Ensures uniform API across pricing models so that they can be 
      seamlessly swapped within the OptionEng class.
    """
    @abc.abstractmethod
    def price(self, option):
        pass
    
    @abc.abstractmethod
    def d1(self, option):
        pass
    
    @abc.abstractmethod
    def d2(self, option):
        pass
    
    @abc.abstractmethod
    def delta(self, option):
        pass
    
    @abc.abstractmethod
    def gamma(self, option):
        pass
    
    @abc.abstractmethod
    def dollarGamma(self, option):
        pass
    
    @abc.abstractmethod
    def theta(self, option):
        pass
    
    @abc.abstractmethod
    def vega(self, option):
        pass
    
    @abc.abstractmethod
    def vanna(self, option):
        pass
    
class BlackScholesEngine(PricingEngine):
    """
    Black-Scholes pricing engine for European options.

    This engine returns closed-form solutions under the Black-Scholes model,
    providing methods for pricing and computing Greeks (Delta, Gamma, Theta, Vega, Vanna, Dollar Gamma).
    
    It requires an Option object with attributes for spot price, strike price, volatility, risk-free rate, maturity, and expiry date.

    Methods
    -------
    d1(option) : np.ndarray
        Compute the Black-Scholes d1 term for the given option.
    d2(option) : np.ndarray
        Compute the Black-Scholes d2 term for the given option.
    price(option) : np.ndarray
        Compute the fair price of the option under the Black-Scholes model.
    delta(option) : np.ndarray
        Compute the option delta.
    gamma(option) : np.ndarray
        Compute the option gamma.
    dollarGamma(option) : np.ndarray
        Compute the option dollar gamma = 0.5 * gamma * S^2.
    theta(option) : np.ndarray
        Compute the option theta (time decay).
    vega(option) : np.ndarray
        Compute the option vega (sensitivity to volatility).
    vanna(option) : np.ndarray
        Compute the option vanna (cross sensitivity between spot and volatility).

    Notes
    -----
    - Only European-style options are supported.
    - Input arrays are fully vectorized, allowing simultaneous pricing of multiple options.
    - Expired options are handled explicitly to mitigate issues with limiting values of closed-form solutions.
    """
    def d1(self, option):
        tau = option.tau()
        S = option.S
        K = option.K
        r = option.r
        sigma = option.sigma
        
        out = np.full_like(S, np.nan, dtype=float)
        live = tau > 0
        if np.any(live):
            out[live] = ((np.log(S[live] / K[live]) + (r[live] + 0.5 * sigma[live]**2) * tau[live]) / (sigma[live] * np.sqrt(tau[live])))
        return out
    
    def d2(self, option):
        tau = option.tau()
        out = np.full_like(option.S, np.nan, dtype=float)
        live = tau > 0
        if np.any(live):
            out[live] = self.d1(option)[live] - option.sigma[live] * np.sqrt(tau[live])
        return out    
    
    def price(self, option):
        tau = option.tau()
        S = option.S
        K = option.K
        r = option.r
        sigma = option.sigma
        type = option.type
        
        out = np.empty_like(S, dtype=float)
        
        expired = tau == 0
        live = ~expired
                
        if np.any(live):
            d1_val = self.d1(option)[live]
            d2_val = self.d2(option)[live]
            if type == 'call':
                out[live] = S[live] * st.norm.cdf(d1_val) - K[live] * np.exp(-r[live] * tau[live]) * st.norm.cdf(d2_val)
            elif type == 'put':
                out[live] = K[live] * np.exp(-r[live] * tau[live]) * st.norm.cdf(-d2_val) - S[live] * st.norm.cdf(-d1_val)
            else:
                raise ValueError("Invalid option type")
        
        if np.any(expired):
            if type == 'call':
                out[expired] = np.maximum(S[expired] - K[expired], 0.0)
            elif type == 'put':
                out[expired] = np.maximum(K[expired] - S[expired], 0.0)
            else:
                raise ValueError("Invalid option type")
        
        return out

    def delta(self, option):
        tau = option.tau()
        S = option.S
        K = option.K
        r = option.r
        sigma = option.sigma
        type = option.type
        
        out = np.empty_like(S, dtype=float)
        
        expired = tau == 0
        live = ~expired
                
        if np.any(expired):
            if type == 'call':
                out[expired] = (S[expired] > K[expired]).astype(float)
            elif type == 'put':
                out[expired] = -(S[expired] < K[expired]).astype(float)
            else:
                raise ValueError("Invalid option type")
        
        if np.any(live):
            d1_val = self.d1(option)[live]
            if type == 'call':
                out[live] = st.norm.cdf(d1_val)
            elif type == 'put':
                out[live] = st.norm.cdf(d1_val) - 1.0
            else:
                raise ValueError("Invalid option type")
        
        return out

    def gamma(self, option):
        tau = np.maximum(option.T - option.t, 0.0)
        S = option.S
        sigma = option.sigma
        
        out = np.zeros_like(S, dtype=float)
        live = tau > 0
                
        if np.any(live):
            d1_val = self.d1(option)[live]
            out[live] = st.norm.pdf(d1_val) / (S[live] * sigma[live] * np.sqrt(tau[live]))
        
        return out

    def dollarGamma(self, option):
        return 0.5 * self.gamma(option) * (option.S ** 2)

    def theta(self, option):
        tau = np.maximum(option.T - option.t, 0.0)
        S = option.S
        K = option.K
        r = option.r
        sigma = option.sigma
        type = option.type
        
        out = np.zeros_like(S, dtype=float)
        live = tau > 0
            
        if np.any(live):
            d1_val = self.d1(option)[live]
            d2_val = self.d2(option)[live]
            if type == 'call':
                out[live] = (- (S[live] * st.norm.pdf(d1_val) * sigma[live]) / (2 * np.sqrt(tau[live]))
                             - r[live] * K[live] * np.exp(-r[live] * tau[live]) * st.norm.cdf(d2_val))
            elif type == 'put':
                out[live] = (- (S[live] * st.norm.pdf(d1_val) * sigma[live]) / (2 * np.sqrt(tau[live]))
                             + r[live] * K[live] * np.exp(-r[live] * tau[live]) * st.norm.cdf(-d2_val))
            else:
                raise ValueError("Invalid option type")
        
        return out

    def vega(self, option):
        tau = option.tau()
        S = option.S
        sigma = option.sigma
        
        out = np.zeros_like(S, dtype=float)
        live = tau > 0
                
        if np.any(live):
            d1_val = self.d1(option)[live]
            out[live] = st.norm.pdf(d1_val) * S[live] * np.sqrt(tau[live])
        
        return out

    def vanna(self, option):
        tau = np.maximum(option.T - option.t, 0.0)
        S = option.S
        sigma = option.sigma
        
        out = np.zeros_like(S, dtype=float)
        live = tau > 0
                
        if np.any(live):
            d1_val = self.d1(option)[live]
            d2_val = self.d2(option)[live]
            out[live] = st.norm.pdf(d1_val) * d2_val / sigma[live]
        
        return out

class BinomialTreeEngine(PricingEngine):
    def price(self, option):
        pass

class MertonJumpDiffusionEngine(PricingEngine):
    def __init__(self, lambda0, alpha, deltaJ, n=50):
        """
        Merton Jump-Diffusion pricing engine.

        Parameters
        ----------
        lambda0 : float
            Jump intensity (expected number of jumps per unit time).
        alpha : float
            Mean of jump size (log-space).
        delta : float
            Standard deviation of jump size (log-space).
        n : int, optional
            Number of jump terms in series expansion.
        """
        self.lambda0 = lambda0
        self.alpha = alpha
        self.deltaJ = deltaJ
        self.n = n

    def d1(self, option):
        raise(ValueError('NaN'))
    
    def d2(self, option):
        raise(ValueError('NaN'))
    
    def price(self, option):
        tau = option.tau()
        sigma = option.sigma
        S = option.S
        type = option.type
        T, t, K, r = option.T, option.t, option.K, option.r
        
        out = np.empty_like(S, dtype=float)
        
        expired = tau == 0
        live = ~expired
                        
        if np.any(live):
            optBS = np.zeros((self.n, len(S[live])))
            lambdabar = self.lambda0 * np.exp(self.alpha + 0.5 * self.deltaJ**2)            
            if type == 'call':
                for j in range(self.n):
                    sigma_j = np.sqrt(sigma[live]**2 + j * self.deltaJ**2 / tau[live])
                    r_j = r[live] - self.lambda0 * (np.exp(self.alpha + 0.5 * self.deltaJ**2) - 1) \
                    + (j * self.alpha + 0.5 * j * self.deltaJ**2) / tau[live]
                    weight = np.exp(-lambdabar * tau[live]) * ((lambdabar * tau[live]) ** j) / m.factorial(j)
                    optBS[j, :] = weight * BlackScholesEngine().price(Option(S[live], K[live], r_j, T[live], t[live], sigma_j, "European", "call"))
                out[live] = np.sum(optBS, axis = 0)
            elif type == 'put':
                for j in range(self.n):
                    sigma_j = np.sqrt(sigma[live]**2 + j * self.deltaJ**2 / tau[live])
                    r_j = r[live] - self.lambda0 * (np.exp(self.alpha + 0.5 * self.deltaJ**2) - 1) \
                    + (j * self.alpha + 0.5 * j * self.deltaJ**2) / tau[live]
                    weight = np.exp(-lambdabar * tau[live]) * ((lambdabar * tau[live]) ** j) / m.factorial(j)
                    optBS[j, :] = weight * BlackScholesEngine().price(Option(S[live], K[live], r_j, T[live], t[live], sigma_j, "European", "put"))
                out[live] = np.sum(optBS, axis = 0)
            else:
                raise ValueError("Invalid option type")
        
        if np.any(expired):
            if type == 'call':
                out[expired] = np.maximum(S[expired] - K[expired], 0.0)
            elif type == 'put':
                out[expired] = np.maximum(K[expired] - S[expired], 0.0)
            else:
                raise ValueError("Invalid option type")
        
        return out
        
    def delta(self, option):
        tau = option.tau()
        type = option.type
        S, T, t, K, r, sigma = option.S, option.T, option.t, option.K, option.r, option.sigma
        lambda0, alpha, deltaJ, n = self.lambda0, self.alpha, self.deltaJ, self.n

        out = np.empty_like(S, dtype=float)
        
        expired = tau == 0
        live = ~expired
        
        if np.any(expired):
            if type == 'call':
                out[expired] = (S[expired] > K[expired]).astype(float)
            elif type == 'put':
                out[expired] = -(S[expired] < K[expired]).astype(float)
            else:
                raise ValueError("Invalid option type")

        if np.any(live):
            delta_BS = np.zeros((self.n, len(S[live])))
            lambdabar = lambda0 * np.exp(alpha + (deltaJ ** 2) / 2)
            if type == 'call':
                for j in range(self.n):
                    sigma_j = np.sqrt(sigma[live]**2 + j * self.deltaJ**2 / tau[live])
                    r_j = r[live] - self.lambda0 * (np.exp(self.alpha + 0.5 * self.deltaJ**2) - 1) \
                    + (j * self.alpha + 0.5 * j * self.deltaJ**2) / tau[live]
                    d1 = OptionEng(S[live], K[live], r_j, T[live], t[live], sigma_j, 'European','call',BlackScholesEngine()).d1()
                    weight = np.exp(-lambda0 * tau[live]) * (lambda0 * tau[live])**j / m.factorial(j)
                    delta_BS[j, :] = weight * st.norm.cdf(d1)
                out[live] = np.sum(delta_BS, axis = 0)
            elif type == 'put':
                for j in range(self.n):
                    sigma_j = np.sqrt(sigma[live]**2 + j * self.deltaJ**2 / tau[live])
                    r_j = r[live] - self.lambda0 * (np.exp(self.alpha + 0.5 * self.deltaJ**2) - 1) \
                    + (j * self.alpha + 0.5 * j * self.deltaJ**2) / tau[live]
                    d1 = OptionEng(S[live], K[live], r_j, T[live], t[live], sigma_j, 'European','call',BlackScholesEngine()).d1()
                    weight = np.exp(-lambda0 * tau[live]) * (lambda0 * tau[live])**j / m.factorial(j)
                    delta_BS[j, :] = weight * (st.norm.cdf(d1) - 1)
                out[live] = np.sum(delta_BS, axis = 0)
            else:
                raise ValueError("Invalid option type")
        
        return out

    def gamma(self, option):
        raise NotImplementedError()
    
    def dollarGamma(self, option):
        raise NotImplementedError()
    
    def theta(self, option):
        raise NotImplementedError()
    
    def vega(self, option):
        raise NotImplementedError()
    
    def vanna(self, option):
        raise NotImplementedError()
    
# ------------------------------------ END -------------------------------------