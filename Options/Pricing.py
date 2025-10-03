import numpy as np
import scipy.stats as st
import math as m
import abc

import Assets as ast

# -------------------------- 1. Pricing Engine Class ---------------------------
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
    def d1(self, option):
        raise(ValueError('NaN: d1 is unique to Black-Scholes'))
    
    def d2(self, option):
        raise(ValueError('NaN: d2 is unique to Black-Scholes'))

    def price(self, option):
        raise NotImplementedError()

class MertonJumpDiffusionEngine(PricingEngine):
    def __init__(self, lam, alpha, deltaJ, n=50):
        """
        Merton Jump-Diffusion pricing engine.

        Parameters
        ----------
        lam : float
            Jump intensity (expected number of jumps per unit time).
        alpha : float
            Mean of jump size (log-space).
        delta : float
            Standard deviation of jump size (log-space).
        n : int, optional
            Number of jump terms in series expansion.
        """
        self.lam = lam
        self.alpha = alpha
        self.deltaJ = deltaJ
        self.n = n

    def d1(self, option):
        raise(ValueError('NaN: d1 is unique to Black-Scholes'))
    
    def d2(self, option):
        raise(ValueError('NaN: d2 is unique to Black-Scholes'))
    
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
            lambdabar = self.lam * np.exp(self.alpha + 0.5 * self.deltaJ**2)            
            if type == 'call':
                for j in range(self.n):
                    sigma_j = np.sqrt(sigma[live]**2 + j * self.deltaJ**2 / tau[live])
                    r_j = r[live] - self.lam * (np.exp(self.alpha + 0.5 * self.deltaJ**2) - 1) \
                    + (j * self.alpha + 0.5 * j * self.deltaJ**2) / tau[live]
                    weight = np.exp(-lambdabar * tau[live]) * ((lambdabar * tau[live]) ** j) / m.factorial(j)
                    optBS[j, :] = weight * BlackScholesEngine().price(ast.OptionEng(S[live], K[live], r_j, T[live], t[live], sigma_j, "European", "call", BlackScholesEngine()))
                out[live] = np.sum(optBS, axis = 0)
            elif type == 'put':
                for j in range(self.n):
                    sigma_j = np.sqrt(sigma[live]**2 + j * self.deltaJ**2 / tau[live])
                    r_j = r[live] - self.lam * (np.exp(self.alpha + 0.5 * self.deltaJ**2) - 1) \
                    + (j * self.alpha + 0.5 * j * self.deltaJ**2) / tau[live]
                    weight = np.exp(-lambdabar * tau[live]) * ((lambdabar * tau[live]) ** j) / m.factorial(j)
                    optBS[j, :] = weight * BlackScholesEngine().price(ast.OptionEng(S[live], K[live], r_j, T[live], t[live], sigma_j, "European", "put", BlackScholesEngine()))
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
        lam, alpha, deltaJ, n = self.lam, self.alpha, self.deltaJ, self.n

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
            lambdabar = lam * np.exp(alpha + (deltaJ ** 2) / 2)
            if type == 'call':
                for j in range(self.n):
                    sigma_j = np.sqrt(sigma[live]**2 + j * self.deltaJ**2 / tau[live])
                    r_j = r[live] - self.lam * (np.exp(self.alpha + 0.5 * self.deltaJ**2) - 1) \
                    + (j * self.alpha + 0.5 * j * self.deltaJ**2) / tau[live]
                    d1 = ast.OptionEng(S[live], K[live], r_j, T[live], t[live], sigma_j, 'European','call',BlackScholesEngine()).d1()
                    weight = np.exp(-lam * tau[live]) * (lam * tau[live])**j / m.factorial(j)
                    delta_BS[j, :] = weight * st.norm.cdf(d1)
                out[live] = np.sum(delta_BS, axis = 0)
            elif type == 'put':
                for j in range(self.n):
                    sigma_j = np.sqrt(sigma[live]**2 + j * self.deltaJ**2 / tau[live])
                    r_j = r[live] - self.lam * (np.exp(self.alpha + 0.5 * self.deltaJ**2) - 1) \
                    + (j * self.alpha + 0.5 * j * self.deltaJ**2) / tau[live]
                    d1 = ast.OptionEng(S[live], K[live], r_j, T[live], t[live], sigma_j, 'European','call',BlackScholesEngine()).d1()
                    weight = np.exp(-lam * tau[live]) * (lam * tau[live])**j / m.factorial(j)
                    delta_BS[j, :] = weight * (st.norm.cdf(d1) - 1)
                out[live] = np.sum(delta_BS, axis = 0)
            else:
                raise ValueError("Invalid option type")
        
        return out

    def gamma(self, option):
        tau = option.tau()
        type = option.type
        S, T, t, K, r, sigma = option.S, option.T, option.t, option.K, option.r, option.sigma
        lam, alpha, deltaJ, n = self.lam, self.alpha, self.deltaJ, self.n

        out = np.zeros_like(S, dtype=float)
        expired = tau == 0
        live = ~expired

        if np.any(live):
            gamma_BS = np.zeros((self.n, len(S[live])))
            for j in range(self.n):
                sigma_j = np.sqrt(sigma[live]**2 + j * deltaJ**2 / tau[live])
                r_j = r[live] - lam * (np.exp(alpha + 0.5 * deltaJ**2) - 1) \
                    + (j * alpha + 0.5 * j * deltaJ**2) / tau[live]
                opteng = ast.OptionEng(S[live], K[live], r_j, T[live], t[live], sigma_j, 'European', type, BlackScholesEngine())
                weight = np.exp(-lam * tau[live]) * (lam * tau[live])**j / m.factorial(j)
                gamma_BS[j, :] = weight * opteng.gamma()
            out[live] = np.sum(gamma_BS, axis=0)
        # gamma is 0 for expired options
        return out

    def dollarGamma(self, option):
        tau = option.tau()
        type = option.type
        S, T, t, K, r, sigma = option.S, option.T, option.t, option.K, option.r, option.sigma
        lam, alpha, deltaJ, n = self.lam, self.alpha, self.deltaJ, self.n

        out = np.zeros_like(S, dtype=float)
        expired = tau == 0
        live = ~expired

        if np.any(live):
            dgamma_BS = np.zeros((self.n, len(S[live])))
            for j in range(self.n):
                sigma_j = np.sqrt(sigma[live]**2 + j * deltaJ**2 / tau[live])
                r_j = r[live] - lam * (np.exp(alpha + 0.5 * deltaJ**2) - 1) \
                    + (j * alpha + 0.5 * j * deltaJ**2) / tau[live]
                opteng = ast.OptionEng(S[live], K[live], r_j, T[live], t[live], sigma_j, 'European', type, BlackScholesEngine())
                weight = np.exp(-lam * tau[live]) * (lam * tau[live])**j / m.factorial(j)
                dgamma_BS[j, :] = weight * opteng.dollarGamma()
            out[live] = np.sum(dgamma_BS, axis=0)
        # dollar gamma is 0 for expired options
        return out

    def theta(self, option):
        tau = option.tau()
        type = option.type
        S, T, t, K, r, sigma = option.S, option.T, option.t, option.K, option.r, option.sigma
        lam, alpha, deltaJ, n = self.lam, self.alpha, self.deltaJ, self.n

        out = np.zeros_like(S, dtype=float)
        expired = tau == 0
        live = ~expired

        if np.any(live):
            theta_BS = np.zeros((self.n, len(S[live])))
            for j in range(self.n):
                sigma_j = np.sqrt(sigma[live]**2 + j * deltaJ**2 / tau[live])
                r_j = r[live] - lam * (np.exp(alpha + 0.5 * deltaJ**2) - 1) \
                    + (j * alpha + 0.5 * j * deltaJ**2) / tau[live]
                opteng = ast.OptionEng(S[live], K[live], r_j, T[live], t[live], sigma_j, 'European', type, BlackScholesEngine())
                weight = np.exp(-lam * tau[live]) * (lam * tau[live])**j / m.factorial(j)
                theta_BS[j, :] = weight * opteng.theta()
            out[live] = np.sum(theta_BS, axis=0)
        # theta is 0 for expired options
        return out

    def vega(self, option):
        tau = option.tau()
        type = option.type
        S, T, t, K, r, sigma = option.S, option.T, option.t, option.K, option.r, option.sigma
        lam, alpha, deltaJ, n = self.lam, self.alpha, self.deltaJ, self.n

        out = np.zeros_like(S, dtype=float)
        expired = tau == 0
        live = ~expired

        if np.any(live):
            vega_BS = np.zeros((self.n, len(S[live])))
            for j in range(self.n):
                sigma_j = np.sqrt(sigma[live]**2 + j * deltaJ**2 / tau[live])
                r_j = r[live] - lam * (np.exp(alpha + 0.5 * deltaJ**2) - 1) \
                    + (j * alpha + 0.5 * j * deltaJ**2) / tau[live]
                opteng = ast.OptionEng(S[live], K[live], r_j, T[live], t[live], sigma_j, 'European', type, BlackScholesEngine())
                weight = np.exp(-lam * tau[live]) * (lam * tau[live])**j / m.factorial(j)
                vega_BS[j, :] = weight * opteng.vega()
            out[live] = np.sum(vega_BS, axis=0)
        # vega is 0 for expired options
        return out

    def vanna(self, option):
        tau = option.tau()
        type = option.type
        S, T, t, K, r, sigma = option.S, option.T, option.t, option.K, option.r, option.sigma
        lam, alpha, deltaJ, n = self.lam, self.alpha, self.deltaJ, self.n

        out = np.zeros_like(S, dtype=float)
        expired = tau == 0
        live = ~expired

        if np.any(live):
            vanna_BS = np.zeros((self.n, len(S[live])))
            for j in range(self.n):
                sigma_j = np.sqrt(sigma[live]**2 + j * deltaJ**2 / tau[live])
                r_j = r[live] - lam * (np.exp(alpha + 0.5 * deltaJ**2) - 1) \
                    + (j * alpha + 0.5 * j * deltaJ**2) / tau[live]
                opteng = ast.OptionEng(S[live], K[live], r_j, T[live], t[live], sigma_j, 'European', type, BlackScholesEngine())
                weight = np.exp(-lam * tau[live]) * (lam * tau[live])**j / m.factorial(j)
                vanna_BS[j, :] = weight * opteng.vanna()
            out[live] = np.sum(vanna_BS, axis=0)
        # vanna is 0 for expired options
        return out
    
# ------------------------------ END ------------------------------