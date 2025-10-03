import numpy as np
import matplotlib.pyplot as plt

import Assets as op

# -------------------- 1. Binomial Tree Backward Induction ---------------------
def OptionEurPrice(S, K, r, sigma, T, t, type, N):
    """Prices European option by CRR parametrisation of binomial tree model.
    
        Recursive tree by backward induction.

    Args:
        S (float): Spot price
        K (float): Strike price
        r (float): Risk-free rate
        sigma (float): Volatility
        T (float): Expiry time
        t (float): Time
        type (str): Option type: 'call'/'put'
        N (int): Partitions of (T-t) interval.

    Raises:
        ValueError: If Option type is invalid.

    Returns:
        float: Option price
    """
    deltaT = (T - t)/ N
    u = np.exp(sigma * np.sqrt(deltaT))
    d = 1.0 / u
    Q = (np.exp(r * deltaT) - d) / (u - d)
    v = np.exp(-r * deltaT)
    
    ST = [S * (u ** j) * (d ** (N - j)) for j in range(N+1)]
    
    if type == 'call':
        V = [max(s - K, 0) for s in ST]
    elif type == 'put':
        V = [max(K - s, 0) for s in ST]
    else:
        raise ValueError('Option Type not call/put')
    
    for n in range(N - 1, -1, -1):
        for j in range(n + 1):            
            V[j] = v * (Q * V[j + 1] + (1 - Q) * V[j])
                        
    return V[0]

def OptionAmPrice(S, K, r, sigma, T, t, type, N):
    """Prices American option by CRR parametrisation of binomial tree model.
    
        Recursive tree by backward induction.

    Args:
        S (float): Spot price
        K (float): Strike price
        r (float): Risk-free rate
        sigma (float): Volatility
        T (float): Expiry time
        t (float): Time
        type (str): Option type: 'call'/'put'
        N (int): Partitions of (T-t) interval.

    Raises:
        ValueError: If Option type is invalid.

    Returns:
        float: Option price
    """
    deltaT = (T - t)/ N
    u = np.exp(sigma * np.sqrt(deltaT))
    d = 1.0 / u
    Q = (np.exp(r * deltaT) - d) / (u - d)
    v = np.exp(-r * deltaT)
    
    ST = [S * (u ** j) * (d ** (N - j)) for j in range(N+1)]
    
    if type == 'call':
        V = [max(s - K, 0) for s in ST]
    elif type == 'put':
        V = [max(K - s, 0) for s in ST]
    else:
        raise ValueError('Option Type not call/put')
    
    for n in range(N - 1, -1, -1):
        for j in range(n + 1):
            St = S * (u ** j) * (d ** (n - j))
            
            CV = v * (Q * V[j + 1] + (1 - Q) * V[j])
            EV = max(St - K, 0) if type == "call" else max(K - St, 0)
            
            V[j] = max(CV, EV)
            
    return V[0]

# ---------------- 2. Accelerated Tree Backward Induction (II) -----------------
def OptionEurPrice_1(S, K, r, sigma, T, t, type, N):
    """Prices European option by CRR parametrisation of binomial tree model. 
    
        Faster algorithm - vectorised backward induction.
    
    Args:
        S (float): Spot price
        K (float): Strike price
        r (float): Risk-free rate
        sigma (float): Volatility
        T (float): Expiry time
        t (float): Time
        type (str): Option type: 'call'/'put'
        N (int): Partitions of (T-t) interval.

    Raises:
        ValueError: If Option type is invalid.

    Returns:
        float: Option price
    """
    deltaT = (T - t)/ N
    u = np.exp(sigma * np.sqrt(deltaT))
    d = 1.0 / u
    Q = (np.exp(r * deltaT) - d) / (u - d)
    v = np.exp(-r * deltaT)
    
    j = np.arange(N + 1)
    ST = S * u ** (2 * j - N)
    
    if type == 'call':
        V = np.maximum(ST - K, 0.0)
    elif type == 'put':
        V = np.maximum(K - ST, 0.0)
    else:
        raise ValueError('Option Type not call/put')
    
    for n in range(N - 1, -1, -1):            
        V = v * (Q * V[1: n + 2] + (1 - Q) * V[0: n + 1])
                        
    return V[0]

def OptionAmPrice_1(S, K, r, sigma, T, t, type, N):
    """Prices American option by CRR parametrisation of binomial tree model. 
    
        Faster algorithm - vectorised backward induction.


    Args:
        S (float): Spot price
        K (float): Strike price
        r (float): Risk-free rate
        sigma (float): Volatility
        T (float): Expiry time
        t (float): Time
        type (str): Option type: 'call'/'put'
        N (int): Partitions of (T-t) interval.

    Raises:
        ValueError: If Option type is invalid.

    Returns:
        float: Option price
    """
    deltaT = (T - t)/ N
    u = np.exp(sigma * np.sqrt(deltaT))
    d = 1.0 / u
    Q = (np.exp(r * deltaT) - d) / (u - d)
    v = np.exp(-r * deltaT)
    
    j = np.arange(N + 1)
    ST = S * u ** (2 * j - N)
    
    if type == 'call':
        V = np.maximum(ST - K, 0.0)
    elif type == 'put':
        V = np.maximum(K - ST, 0.0)
    else:
        raise ValueError('Option Type not call/put')
    
    for n in range(N - 1, -1, -1):
        CV = v * (Q * V[1: n + 2] + (1 - Q) * V[0: n + 1])
        
        j = np.arange(n + 1)        
        St = S * (u ** j) * (d ** (n - j))
        
        if type == 'call':
            EV = np.maximum(St - K, 0.0)
        else:
            EV = np.maximum(K - St, 0.0)

        V = np.maximum(CV, EV)

    return V[0]

# ------------------------------------ END -------------------------------------