import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import math as m

import Assets as ast
import MonteCarlo as mc
import Pricing as pr

# ------------------------ 1. Black-Scholes Simulation ------------------------
def BSsim0(r, mu, sigma, S0, K, type, dt):
    """
    Simulates a geometric Brownian motion stock price path and computes
    the corresponding Black-Scholes option metrics at each time step.
    
    Samples paths by fastest algo for single draws. Not optmised for speed but rather for displaying Greeks & realised volatility metrics for analysis.
    
    Parameters
    ----------
    r : float
        Risk-free interest rate.
    mu : float
        Drift of the price.
    sigma : float
        Volatility of spot price.
    S0 : float
        Initial spot price.
    K : float
        Strike price.
    type : str
        Type of option: 'call'/'put'.
    dt : float
        Time step size (fraction of year).

    Returns
    -------
    tuple
    
    t : list of float
        Time step value (fraction of year).
    spot : ndarray
        Simulated GBM spot price path.
    option_price : ndarray
        Black-Scholes option prices at each time step.
    delta : ndarray
        Option delta at each time step.
    gamma : ndarray
        Option gamma at each time step.
    dollar_gamma : ndarray
        Option dollar gamma at each time step.
    realised_vol : ndarray
        Rolling realised volatility of the underlying.
    theta : ndarray
        Option theta at each time step.
    """
    N = int(1 / dt)
    t = [j / N for j in range(N + 1)]
    spot = mc.gbmP0(mu, sigma, S0, 1, dt)
    underlying = ast.Spot(spot,r, t, 1, dt)
    option = ast.OptionEng(spot, K, r, 1, t, sigma, 'European', type, pr.BlackScholesEngine(), dt)
    scal_vol = underlying.VarReal()
    return t, spot, option.price(), option.delta(), option.gamma(), option.dollarGamma(), scal_vol, option.theta()

def BSsim1(r, mu, sigma, S0, K, type, dt):
    """
    Simulates a geometric Brownian motion stock price path and computes
    the corresponding Black-Scholes option metrics at each time step.
    
    Optimised for speed rather than visualisation.

    Parameters
    ----------
    r : float
        Risk-free interest rate.
    mu : float
        Drift of the stock price.
    sigma : float
        Volatility of the stock price.
    S0 : float
        Initial stock price.
    dt : float
        Time step size (fraction of year, e.g., 0.01).
    K : float
        Strike price of the call option.
    R : int, optional
        Number of rebalancing periods per year (not used directly here).

    Returns
    -------
    tuple
    
    t : list of float
        Time steps from 0 to 1.
    spot : ndarray
        Simulated GBM stock price path.
    option_price : ndarray
        Black-Scholes option prices at each time step.
    delta : ndarray
        Option delta at each time step.
    gamma : ndarray
        Option gamma at each time step.
    dollar_gamma : ndarray
        Option dollar gamma at each time step.
    realised_vol : ndarray
        Rolling realised volatility of the underlying.
    theta : ndarray
        Option theta at each time step.
    """
    N = int(1 / dt)
    t = [j / N for j in range(N + 1)]
    spot = mc.gbmP0(mu, sigma, S0, 1, dt)
    option = ast.OptionEng(spot.T, K, r, 1, t, sigma, 'European', type, pr.BlackScholesEngine(), dt)
    return t, spot.T, option.price(), option.delta()

def BSsim2(r, mu, sigma, S0, K, type, dt, paths):
    """
    Simulates a geometric Brownian motion stock price path and computes
    the corresponding Black-Scholes option metrics at each time step.
    
    Jointly samples paths, collecting into matrix with paths (rows) x 1/dt (cols). Then collects option Greeks into same structure. 
    
    Efficient when drawing large numbers of paths for Monte Carlo.

    Parameters
    ----------
    r : float
        Risk-free interest rate.
    mu : float
        Drift of the stock price.
    sigma : float
        Volatility of the stock price.
    S0 : float
        Initial stock price.
    dt : float
        Time step size (fraction of year, e.g., 0.01).
    K : float
        Strike price of the call option.
    R : int, optional
        Number of rebalancing periods per year (not used directly here).

    Returns
    -------
    tuple
    
    t : list of float
        Time steps from 0 to 1.
    spot : ndarray
        Simulated GBM stock price path.
    option_price : ndarray
        Black-Scholes option prices at each time step.
    delta : ndarray
        Option delta at each time step.
    """
    N = int(1 / dt)
    t = [j / N for j in range(N + 1)]
    spot = mc.gbmP2(mu, sigma, S0, 1, dt, paths)
    option = ast.OptionEng(spot, K, r, 1, t, sigma, 'European', type, pr.BlackScholesEngine(), dt)
    return t, spot, option.price(), option.delta()

def BSplot(r, mu, sigma, S0, K, type, dt):
    """
    Plots a panel of subplots for Black-Scholes option metrics over time.

    Parameters
    ----------
    r : float
        Risk-free interest rate.
    mu : float
        Drift of the stock price.
    sigma : float
        Volatility of the stock price.
    S0 : float
        Initial stock price.
    K : float
        Strike price of the option.
    type: str
        Type of option: 'call'/'put'.
    dt : float
        Time step size (fraction of year, e.g., 0.01).

    Returns
    -------
    None
        Displays a figure with a 3x3 panel of subplots showing:
        - Spot price
        - Option price
        - Delta
        - Gamma
        - Dollar Gamma
        - Realised Variance
        - Theta
    """
    t, spot, vBS, deltaBS, gammaBS, dolgammaBS, spotVol, theta = BSsim0(r, mu, sigma, S0, K, type, dt)

    labels = [
        ("Spot Price", spot),
        ("Option Price", vBS),
        ("Delta", deltaBS),
        ("Gamma", gammaBS),
        ("Dollar Gamma", dolgammaBS),
        ("Realised Variance", spotVol),
        ("Theta", theta)
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, (label, data) in zip(axes, labels):
        ax.plot(t, data, '-o', ms=0.8, alpha=0.6, label=label)
        ax.set_title(label)
        ax.set_xlabel("Time")
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    plt.show()
      
# ----------------------------- 2. MJD Simulation ------------------------------
def MJDsim(r, mu, sigma, S0, K, type, dt, lam, alpha, deltaJ, n):
    '''
    MJD option and delta simulation.
    '''
    N = int(1 / dt)
    t = [j / N for j in range(N + 1)]
    spot = mc.MJD(mu, sigma, S0, lam, alpha, deltaJ, dt)
    option = ast.OptionEng(spot, K, r, 1, t, sigma, 'European', type, pr.MertonJumpDiffusionEngine(lam, alpha, deltaJ, n), dt)
    option_BS = ast.OptionEng(spot, K, r, 1, t, sigma, 'European', type, pr.BlackScholesEngine(), dt)
    return [t, spot, option.price(), option.delta(), option_BS.price(), option_BS.delta()]

def MJDplot(r, mu, sigma, S0, K, type, dt, lam, alpha, deltaJ, n):
    '''
    Panel plot of MJD option simulation mirroring BSplot style.
    Plots spot price, option price, and delta over time.
    '''
    t, spot, vMJD, deltaMJD, vBS, deltaBS = MJDsim(r, mu, sigma, S0, K, type, dt, lam, alpha, deltaJ, n)

    labels = [
        ("Spot Price", spot),
        ("Option Price", vMJD),
        ("Delta", deltaMJD),
        ("BS Option Price", vBS),
        ("BS Delta", deltaBS),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (label, data) in zip(axes, labels):
        ax.plot(t, data, '-o', ms=1, alpha=0.6, label=label)
        ax.set_title(label)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True)
        ax.legend()

    for j in range(len(labels), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    
# -------------------- 3. Black-Scholes Trading Simulations --------------------
def dhedge0(r, mu, sigma, S0, K, type, dt, R):
    """
    Simulates a discrete-time delta-hedging strategy under the Black-Scholes model with periodic portfolio rebalancing.
    
    Original loop-based simulation. Not efficient or in use.

    Parameters
    ----------
    r : float
        Risk-free interest rate.
    mu : float
        Drift of the underlying stock.
    sigma : float
        Volatility of the underlying stock.
    S0 : float
        Initial stock price.
    dt : float
        Time step size (fraction of a year, e.g., 0.01).
    K : float
        Strike price of the option.
    R : int
        Number of rebalancing periods per year.

    Returns
    -------
    tuple of ndarray
    
    prof : float
        Nominal profit of the hedging portfolio over time.
    prof_adj : float
        Discounted and normalized profit over time.
    valS : float
        Value of stock holdings over time.
    valB : float
        Value of bond holdings over time.
    qS : float
        Number of shares held over time.
    qB : float
        Bond holdings over time.
    B : float
        Bond price process over time.
    S : float
        Underlying stock price path.
    V : float
        Option price path.
    delta : float 
        Option delta over time.
    gamma : float
        Option gamma over time.
    dolGamma : float
        Option dollar gamma over time.
    vol : float
        Realized volatility of the underlying.
    theta : float
        Option theta over time.
    time : float
        Time (yrs).
    """
    time, S, V, delta, gamma, dolGamma, vol, theta = BSsim0(r, mu, sigma, S0, K, type, dt)
    
    N = len(time)
    reb = int(N / R)
    qS = np.zeros(N)
    qB = np.zeros(N)
    prof = np.zeros(N)
    prof_adj = np.zeros(N)
    vS = np.zeros(N)
    vB = np.zeros(N)
    B = np.zeros(N)
    B[0] = 1
    qS[0] = delta[0]
    qB[0] = V[0] - delta[0] * S[0]
    vS[0] = qS[0] * S[0]
    vB[0] = qB[0] * B[0]
    prof[0] = vS[0] + vB[0] - V[0]
    
    for t in range(1, N):
        if t % reb == 0:
            B[t] = B[t-1] * np.exp(r * dt)
            qS[t] = delta[t]
            qB[t] = qB[t-1] - S[t] * (delta[t] - delta[t-reb]) / B[t]
            vS[t] = qS[t] * S[t]
            vB[t] = qB[t] * B[t]
            prof[t] = vS[t] + vB[t] - V[t]
            prof_adj[t] = np.exp(-r * dt * t) * prof[t] / V[0]
        else:
            B[t] = B[t-1] * np.exp(r * dt)
            qS[t] = qS[t-1]
            qB[t] = qB[t-1]
            vS[t] = qS[t] * S[t]
            vB[t] = qB[t] * B[t] 
            prof[t] = vS[t] + vB[t] - V[t]
            prof_adj[t] = np.exp(-r * dt * t) * prof[t] / V[0]
    return prof, prof_adj, vS, vB, qS, qB, B, S, V, delta, gamma, dolGamma, vol, theta, time

def dhedge1(r, mu, sigma, S0, K, type, dt, R):
    """
    Simulates a discrete-time delta-hedging strategy under the Black-Scholes model with periodic portfolio rebalancing.
    
    Vectorized algorithmic structure replaces loops. Fasted diffusion path algo but not optimised for overall speed.
    
    Returns numerous Greeks for use in plotting function.
    """
    time, S, V, delta, gamma, dolGamma, vol, theta = BSsim0(r, mu, sigma, S0, K, type, dt)
    N = len(time)
    reb = int(N / R)
    idx = np.arange(N) #effective time vector

    B = np.exp(r * dt * idx)

    rebalance_idx = (idx % reb == 0) & (idx != 0) & (idx != N-1 if (N-1)!=R else idx == idx)

    qS = np.zeros(N)
    qS[0] = delta[0]
    qS[rebalance_idx] = delta[rebalance_idx]
    qS = pd.Series(qS).replace(0, np.nan).ffill().to_numpy()

    qB = np.zeros(N)
    qB[0] = V[0] - qS[0] * S[0]
    trade_cost = np.zeros(N)
    trade_cost[rebalance_idx] = -S[rebalance_idx] * (
        delta[rebalance_idx] - delta[idx[rebalance_idx] - reb]
    )
    qB = qB[0] + np.cumsum(trade_cost / B)

    vS = qS * S
    vB = qB * B
    prof = vS + vB - V
    prof_adj = np.exp(-r * dt * idx) * prof / V[0]

    return prof, prof_adj, vS, vB, qS, qB, B, S, V, delta, gamma, dolGamma, vol, theta, time

def dhedge2(r, mu, sigma, S0, K, type, dt, R):
    """
    Simulates a discrete-time delta-hedging strategy under the Black-Scholes model with periodic portfolio rebalancing.
    
    Vectorized delta-hedging under Black-Scholes.
    
    Drops additional variables for plotting -- optimised for speed rather than visualisation.
    """
    time, S, V, delta = BSsim1(r, mu, sigma, S0, K, type, dt)
    N = len(time)
    reb = int(N / R)

    B = np.exp(r * dt * np.arange(N))

    idx = np.arange(N)
    rebalance_idx = (idx % reb == 0) & (idx != 0) & (idx != N-1 if (N-1)!=R else idx == idx)

    qS = np.zeros(N)
    qS[0] = delta[0]
    qS[rebalance_idx] = delta[rebalance_idx]
    qS = pd.Series(qS).replace(0, np.nan).ffill().to_numpy()

    qB = np.zeros(N)
    qB[0] = V[0] - qS[0] * S[0]
    trade_cost = np.zeros(N)
    trade_cost[rebalance_idx] = -S[rebalance_idx] * (
        delta[rebalance_idx] - delta[idx[rebalance_idx] - reb]
    )
    qB = qB[0] + np.cumsum(trade_cost / B)

    vS = qS * S
    vB = qB * B
    prof = vS + vB - V
    prof_adj = np.exp(-r * dt * idx) * prof / V[0]

    return prof, prof_adj, vS, vB, qS, qB, B, S, V, delta, time

def dhedge3(r, mu, sigma, S0, K, type, dt, R, paths):
    """
    Simulates a discrete-time delta-hedging strategy under the Black-Scholes model with periodic portfolio rebalancing.
    
    Vectorized delta-hedging under Black-Scholes. Handles jointly sampled paths.
    
    Efficient when drawing large numbers of paths for Monte Carlo. Not yet incorporated into analysis functions.
    """
    time, S, V, delta = BSsim2(r, mu, sigma, S0, K, type, dt, paths)
    N = len(time)
    reb = int(N / R)
    idx = np.arange(N) #effective time vector for bond
    tr_idx = np.vstack([np.arange(N) for i in range(paths)])

    B = np.exp(r * dt * idx)

    rebalance_idx = (idx % reb == 0) & (idx != 0) & (idx != N-1) # boolean

    qS = np.zeros((paths, N))
    qS[:,0] = delta[:,0]
    qS[:, rebalance_idx] = delta[:, rebalance_idx]

    qS = np.where(qS != 0, qS, np.nan)        # replace 0 with NaN
    valid = np.where(~np.isnan(qS), idx, 0)
    last_valid = np.maximum.accumulate(valid, axis = 1) # carry forward last index

    rows = np.arange(qS.shape[0])[:, None]
    qS = qS[rows, last_valid]

    qB = np.zeros((paths, N))
    qB[:, 0] = V[:, 0] - qS[:, 0] * S[:, 0]

    # compute changes in stock holdings
    delta_qS = np.zeros_like(qS)
    delta_qS[:, 1:] = qS[:, 1:] - qS[:, :-1]

    # cash required for trades at each step, discounted
    cash_flow = (S * delta_qS) / B[None, :]

    # cumulative financing from trades
    qB = qB[:, [0]] - np.cumsum(cash_flow, axis=1)

    vS = qS * S
    vB = qB * B
    prof = vS + vB - V
    prof_adj = np.exp(-r * dt * idx) * prof / V[:,0][0]

    return prof, prof_adj, vS, vB, qS, qB, B, S, V, delta, time

# ------------------- 4. Black-Scholes Trading Sims Analysis -------------------
def dhedgePlot(r, mu, sigma, S0, K, type, dt, R):
    """
    Creates a 4x4 panel of subplots visualizing the delta-hedging performance
    and option metrics over time for a Black-Scholes option with periodic rebalancing.

    Parameters
    ----------
    r : float
        Risk-free interest rate.
    mu : float
        Drift of the underlying stock.
    sigma : float
        Volatility of the underlying stock.
    S0 : float
        Initial stock price.
    dt : float
        Time step size (fraction of a year, e.g., 0.01).
    K : float
        Strike price of the option.
    R : int
        Number of rebalancing periods per year.

    Returns
    -------
    None
        Displays a figure with 16 subplots showing:
        - Nominal Profit
        - Net Return
        - Value of Stock Portfolio
        - Value of Bond Portfolio
        - Stock Holdings and Delta
        - Bond Holdings
        - Bond Price
        - Stock Price
        - Option Price
        - Gamma
        - Dollar Gamma
        - Realised Variance
        - Gamma Rent
        - Theta
        - Gamma Rent and Theta
        - Gamma Rent less Theta
    """
    prof, prof_adj, vS, vB, qS, qB, B, S, V, delta, gamma, dolGamma, vol, theta, time = dhedge1(r, mu, sigma, S0, K, type, dt, R)
    labels = ['Nominal Profit', 'Net Return', 'Value of Stock Portfolio',       'Value of Bond Portfolio', 'Stock Holdings','Bond Holdings', 'Bond Price', 'Stock Price', 'Option Price', 'Gamma', 'Dollar Gamma', 'Realised Variance (Ann)','GammaRent','Theta','Gamma Rent, Theta','Gamma Rent less Theta','-cumGamRentLessTheta','PDE Gamma Term','PDE int term','PDE Sum']
    fig, axes = plt.subplots(5, 4, figsize=(18, 12))
    axes = axes.flatten()
    t = np.arange(len(prof)+1)
    axes[0].plot(time, prof, '-o', ms=2, alpha=0.6)
    axes[1].plot(time, prof_adj, '-o', ms=2, alpha=0.6)
    axes[2].plot(time, vS, '-o', ms=2, alpha=0.6)
    axes[3].plot(time, vB, '-o', ms=2, alpha=0.6)
    axes[4].plot(time, qS, '-o', ms=2, alpha=0.6)
    axes[4].plot(time, delta, '-o', ms=0.1, alpha=0.6)
    axes[4].legend(['qS','delta'])
    axes[5].plot(time, qB, '-o', ms=2, alpha=0.6)
    axes[6].plot(time, B, '-o', ms=2, alpha=0.6)
    axes[7].plot(time, S, '-o', ms=2, alpha=0.6)
    axes[8].plot(time, V, '-o', ms=2, alpha=0.6)
    axes[9].plot(time, gamma, '-o', ms=2, alpha=0.6)
    axes[10].plot(time, dolGamma, '-o', ms=2, alpha=0.6)
    axes[11].plot(time, vol, '-o', ms=2, alpha=0.6)
    axes[12].plot(time, vol * dolGamma, '-o', ms=2, alpha=0.6)
    axes[13].plot(time, theta, '-o', ms=2, alpha=0.6)
    axes[14].plot(time, vol * dolGamma, '-o', ms=2, alpha=0.6)
    axes[14].plot(time, -theta, '-o', ms=2, alpha=0.6)
    axes[14].legend(['Gamma Rent','Theta'])
    axes[15].plot(time, vol * dolGamma + theta, ms=2, alpha=0.6)
    axes[16].plot(time, (-V[0] / B) * np.cumsum(np.nan_to_num(vol * dolGamma + theta)), ms=2, alpha=0.6)
    axes[17].plot(time, 0.5 * gamma * (S * sigma) ** 2, ms=2, alpha=0.6)
    axes[18].plot(time, r * (delta * S - V), ms=2, alpha=0.6)
    axes[19].plot(time, (0.5 * gamma * (S * sigma) ** 2) + (r * (delta * S - V)) + theta, ms=2, alpha=0.6)

    for j in range(20):
        ax = axes[j]
        ax.set_title(labels[j])
        ax.set_xlabel('Time step')
        ax.set_ylabel('Value')
        ax.grid(True)
    plt.tight_layout()
    plt.show()
    
def dhedgeHist(r, mu, sigma, S0, K, type, dt, R, N):
    """
    Monte Carlo function for simulating trading strategy multiple times, returning the histogram of adjusted profit.
    
    Delta-hedging simulated by dhedge2() to optimise run time.
    
    Parameters
    ----------
    r : float
        Risk-free interest rate.
    mu : float
        Drift of the underlying stock.
    sigma : float
        Volatility of the underlying stock.
    S0 : float
        Initial stock price.
    dt : float
        Time step size (fraction of a year).
    K : float
        Strike price of the option.
    N : int
        Number of simulation runs.
    R : int
        Number of rebalancing periods per year.

    Returns
    -------
    None
        Displays a histogram of the final adjusted profits from N simulations.
    """
    fig, ax = plt.subplots(figsize=[11, 5])
    p = np.zeros(N)
    for j in range(N):
        p[j] = dhedge2(r, mu, sigma, S0, K, type, dt, R)[1][int(1/dt)]
    plt.hist(p, bins=500)
    plt.show()
    
def dhedgeError(r, mu, sigma, S0, K, type, dt, R, N):
    """
    Monte Carlo function for simulating trading strategy multiple times, returning the final adjusted portfolio profits for each simulation.
    
    Delta-hedging simulated by dhedge2() to optimise run time.

    Parameters
    ----------
    r : float
        Risk-free interest rate.
    mu : float
        Drift of the underlying stock.
    sigma : float
        Volatility of the underlying stock.
    S0 : float
        Initial stock price.
    dt : float
        Time step size (fraction of a year).
    K : float
        Strike price of the option.
    R : int
        Number of rebalancing periods per year.
    N : int
        Number of simulation runs.

    Returns
    -------
    ndarray
        Array of length N containing the final adjusted profits from each simulation.
    """
    p = np.zeros(N)
    for j in range(N):
        p[j] = dhedge2(r, mu, sigma, S0, K, type, dt, R)[1][int(1/dt)]
    return p

def dhedgeHistPDF(r, mu, sigma, S0, K, type, dt, R, N):
    """
    Simulates a delta-hedging strategy multiple times and plots a histogram
    and kernel density estimate (KDE) of the final adjusted portfolio profits.
    
    Delta-hedging simulated by dhedge2() to optimise run time.

    Parameters
    ----------
    r : float
        Risk-free interest rate.
    mu : float
        Drift of the underlying stock.
    sigma : float
        Volatility of the underlying stock.
    S0 : float
        Initial stock price.
    dt : float
        Time step size (fraction of a year).
    K : float
        Strike price of the option.
    R : int
        Number of rebalancing periods per year.
    N : int
        Number of simulation runs.

    Returns
    -------
    None
        Displays a figure with a histogram and KDE of the final adjusted profits.
    """
    d = dhedgeError(r, mu, sigma, S0, K, type, dt, R, N)
    print(f"""
    Parameters: r={r}, mu={mu}, sigma={sigma}, S0={S0}, K={K}, dt={dt}, R={R}, N={N}
    Results: min={np.min(d):.4f}, max={np.max(d):.4f}, mean={np.mean(d):.4f}, 
             std={np.std(d, ddof=1):.4f}, kurt={st.kurtosis(d):.4f}, skew={st.skew(d):.4f}
    """)
    fig, ax = plt.subplots(figsize=[11, 5])
    ax.set_xlim([np.mean(d) - np.std(d, ddof=1), np.mean(d) + np.std(d, ddof=1)]) # pyright: ignore[reportArgumentType]
    kde = st.gaussian_kde(d)
    x0 = np.linspace(np.mean(d) - 2 * np.std(d, ddof=1), np.mean(d) + 2 * np.std(d, ddof=1), num=2000)
    k1 = kde(x0)
    plt.hist(d, bins=1000, density=True, color = 'orange', alpha = 0.6, zorder = 1)
    ax.plot(x0, k1, color = 'black', zorder = 2)
    plt.show()

# ------------------------------ BREAK -------------------------------
# ------------------------ 5. MJD Trading Simulations --------------------------
def dhedgeMJD0(r, mu, sigma, S0, K, type, dt, lam, alpha, deltaJ, n, R):
    """
    Simulates a discrete-time delta-hedging strategy under the MJD model with periodic portfolio rebalancing.
    
    Delta calculated according to adjusted delta in Merton (1976) rather than original Black-Scholes.
    
    Original loop-based simulation. Not efficient or in use.
    
    Parameters
    ----------
    r : float
        Risk-free interest rate.
    mu : float
        Drift of the underlying stock.
    sigma : float
        Volatility of the underlying stock.
    S0 : float
        Initial stock price.
    dt : float
        Time step size (fraction of a year, e.g., 0.01).
    K : float
        Strike price of the option.
    R : int
        Number of rebalancing periods per year.

    Returns
    -------
    tuple of ndarray
    
    prof : float
        Nominal profit of the hedging portfolio over time.
    prof_adj : float
        Discounted and normalized profit over time.
    valS : float
        Value of stock holdings over time.
    valB : float
        Value of bond holdings over time.
    qS : float
        Number of shares held over time.
    qB : float
        Bond holdings over time.
    B : float
        Bond price process over time.
    S : float
        Underlying stock price path.
    V : float
        Option price path.
    delta : float 
        Option delta over time.
    gamma : float
        Option gamma over time.
    dolGamma : float
        Option dollar gamma over time.
    vol : float
        Realized volatility of the underlying.
    theta : float
        Option theta over time.
    time : float
        Time (yrs).
    """
    time, S, V, delta, V_BS, delta_BS = MJDsim(r, mu, sigma, S0, K, type, dt, lam, alpha, deltaJ, n)
        
    N = len(time)
    reb = int(N / R)
    qS = np.zeros(N)
    qB = np.zeros(N)
    prof = np.zeros(N)
    prof_adj = np.zeros(N)
    vS = np.zeros(N)
    vB = np.zeros(N)
    B = np.zeros(N)
    B[0] = 1
    qS[0] = delta[0]
    qB[0] = V[0] - delta[0] * S[0]
    vS[0] = qS[0] * S[0]
    vB[0] = qB[0] * B[0]
    prof[0] = vS[0] + vB[0] - V[0]
    
    for t in range(1, N):
        if t % reb == 0:
            B[t] = B[t-1] * np.exp(r * dt)
            qS[t] = delta[t]
            qB[t] = qB[t-1] - S[t] * (delta[t] - delta[t-reb]) / B[t]
            vS[t] = qS[t] * S[t]
            vB[t] = qB[t] * B[t]
            prof[t] = vS[t] + vB[t] - V[t]
            prof_adj[t] = np.exp(-r * dt * t) * prof[t] / V[0]
        else:
            B[t] = B[t-1] * np.exp(r * dt)
            qS[t] = qS[t-1]
            qB[t] = qB[t-1]
            vS[t] = qS[t] * S[t]
            vB[t] = qB[t] * B[t] 
            prof[t] = vS[t] + vB[t] - V[t]
            prof_adj[t] = np.exp(-r * dt * t) * prof[t] / V[0]
            
    return prof, prof_adj, vS, vB, qS, qB, B, S, V, delta, V_BS, delta_BS, time

def dhedgeMJD1(r, mu, sigma, S0, K, type, dt, lam, alpha, deltaJ, n, R):
    """Simulates a discrete-time delta-hedging strategy under the MJD model with periodic portfolio rebalancing.
    
    Delta calculated according to adjusted delta in Merton (1976) rather than original Black-Scholes.
    
    Vectorized algorithmic structure replaces loops. Fasted diffusion path algo.

    Args:
        r (_type_): _description_
        mu (_type_): _description_
        sigma (_type_): _description_
        S0 (_type_): _description_
        K (_type_): _description_
        type (_type_): _description_
        dt (_type_): _description_
        lam (_type_): _description_
        alpha (_type_): _description_
        deltaJ (_type_): _description_
        n (_type_): _description_
        R (_type_): _description_

    Returns:
        _type_: _description_
    """
    time, S, V, delta, V_BS, delta_BS = MJDsim(r, mu, sigma, S0, K, type, dt, lam, alpha, deltaJ, n)
    N = len(time)
    reb = int(N / R)
    idx = np.arange(N) #effective time vector

    B = np.exp(r * dt * idx)

    rebalance_idx = (idx % reb == 0) & (idx != 0) & (idx != N-1 if (N-1)!=R else idx == idx)

    qS = np.zeros(N)
    qS[0] = delta[0]
    qS[rebalance_idx] = delta[rebalance_idx]
    qS = pd.Series(qS).replace(0, np.nan).ffill().to_numpy()

    qB = np.zeros(N)
    qB[0] = V[0] - qS[0] * S[0]
    trade_cost = np.zeros(N)
    trade_cost[rebalance_idx] = -S[rebalance_idx] * (
        delta[rebalance_idx] - delta[idx[rebalance_idx] - reb]
    )
    qB = qB[0] + np.cumsum(trade_cost / B)

    vS = qS * S
    vB = qB * B
    prof = vS + vB - V
    prof_adj = np.exp(-r * dt * idx) * prof / V[0]

    return prof, prof_adj, vS, vB, qS, qB, B, S, V, delta, V_BS, delta_BS, time

# ------------------------------ NOT CLEANED WIP -------------------------------
# ------------------------ 6. MJD Trading Sims Analysis ------------------------
def dhedgeMJDplot(r, mu, sigma, S0, K, type, dt, lam, alpha, deltaJ, n, R):
    """
    MJD plot - Merton delta hedge

    Parameters
    ----------
    r : float
        Risk-free interest rate.
    mu : float
        Drift of the underlying stock.
    sigma : float
        Volatility of the underlying stock.
    S0 : float
        Initial stock price.
    dt : float
        Time step size (fraction of a year, e.g., 0.01).
    K : float
        Strike price of the option.
    R : int
        Number of rebalancing periods per year.

    Returns
    -------
    None
        Displays a figure with 16 subplots showing:
        - Nominal Profit
        - Net Return
        - Value of Stock Portfolio
        - Value of Bond Portfolio
        - Stock Holdings and Delta
        - Bond Holdings
        - Bond Price
        - Stock Price
        - Option Price
        - Gamma
        - Dollar Gamma
        - Realised Variance
        - Gamma Rent
        - Theta
        - Gamma Rent and Theta
        - Gamma Rent less Theta
    """
    prof, prof_adj, valS, valB, qS, qB, B, S, V, delta, V_BS, delta_BS, time = dhedgeMJD1(r, mu, sigma, S0, K, type, dt, lam, alpha, deltaJ, n, R)
    labels = ['Nominal Profit', 'Net Return', 'Value of Stock Portfolio',       'Value of Bond Portfolio', 'Stock Holdings','Bond Holdings', 'Bond Price', 'Stock Price', 'Option Price', 'Delta Comp','V Comp']
    fig, axes = plt.subplots(4, 3, figsize=(18, 12))
    axes = axes.flatten()
    t = np.arange(len(prof)+1)
    axes[0].plot(time, prof, '-o', ms=2, alpha=0.6)
    axes[1].plot(time, prof_adj, '-o', ms=2, alpha=0.6)
    axes[2].plot(time, valS, '-o', ms=2, alpha=0.6)
    axes[3].plot(time, valB, '-o', ms=2, alpha=0.6)
    axes[4].plot(time, qS, '-o', ms=2, alpha=0.6)
    axes[4].plot(time, delta, '-o', ms=0.1, alpha=0.6)
    axes[4].legend(['qS','delta'])
    axes[5].plot(time, qB, '-o', ms=2, alpha=0.6)
    axes[6].plot(time, B, '-o', ms=2, alpha=0.6)
    axes[7].plot(time, S, '-o', ms=2, alpha=0.6)
    axes[8].plot(time, V, '-o', ms=2, alpha=0.6)
    axes[9].plot(time, delta, '-o', ms=2, alpha=0.6)
    axes[9].plot(time, delta_BS, '-o', ms=2, alpha=0.6)
    axes[9].legend(['delta','delta_BS'])
    axes[10].plot(time, V, '-o', ms=2, alpha=0.6)
    axes[10].plot(time, V_BS, '-o', ms=2, alpha=0.6)
    axes[10].legend(['V_MJD','V_BS'])

    for j in range(11):
        ax = axes[j]
        ax.set_title(labels[j])
        ax.set_xlabel('Time step')
        ax.set_ylabel('Value')
        ax.grid(True)
    plt.tight_layout()
    plt.show()
    
def dhedgeMJDhist(r, mu, sigma, S0, K, type, dt, lam, alpha, deltaJ, n, R, N):
    """
    Monte Carlo function for simulating trading strategy multiple times, returning the histogram of adjusted profit.
    
    Mertin (1976) adjusted delta strategy.
    
    Delta-hedging simulated by dhedgeMJD1() to optimise run time.
    """
    fig, ax = plt.subplots(figsize=[11, 5])
    p = np.zeros(N)
    for j in range(N):
        p[j] = dhedgeMJD1(r, mu, sigma, S0, K, type, dt, lam, alpha, deltaJ, n, R)[1][int(1/dt)]
    plt.hist(p, bins=500)
    plt.show()
    
def dhedgeMJDerror(r, mu, sigma, S0, K, type, dt, lam, alpha, deltaJ, n, R, N):
    """
    Monte Carlo function for simulating trading strategy multiple times, returning the final adjusted portfolio profits for each simulation.
    
    Mertin (1976) adjusted delta strategy.
    
    Delta-hedging simulated by dhedgeMJD1() to optimise run time.
    """

    p = np.zeros(N)
    for j in range(N):
        p[j] = dhedgeMJD1(r, mu, sigma, S0, K, type, dt, lam, alpha, deltaJ, n, R)[1][int(1/dt)]
    return p

def dhedgeMJDhistpdf(r, mu, sigma, S0, K, type, dt, lam, alpha, deltaJ, n, R, N):
    d = dhedgeMJDerror(r, mu, sigma, S0, K, type, dt, lam, alpha, deltaJ, n, R, N)
    print("(r, mu, sigma, S0, 1/dt, K, R, N, min, max, mu_eps, sigma_eps, kurtosis_eps, skew_eps)",[r, mu, sigma, S0, 1/dt, K, R, N, np.min(d), np.max(d), np.mean(d),np.std(d,ddof=1), st.kurtosis(d),st.skew(d)])
    fig, ax = plt.subplots(figsize=[11, 5])
    kde = st.gaussian_kde(d)
    x0 = np.linspace(np.min(d), np.max(d), num=1000)
    k1 = kde(x0)
    ax.plot(x0, k1)
    plt.hist(d, bins=500, density=True)
    # tikzplotlib.save("A.tex")
    plt.show()

def plotsJ(r, mu, sigma, S0, K, type, dt, lam, alpha, deltaJ, n, R, N):
    '''
    Final Plots for Jumps.
    '''
    fig, ax = plt.subplots(figsize=[11, 5])
    d = dhedgeMJDerror(r, mu, sigma, S0, K, type, dt, lam, alpha, deltaJ, n, R, N)
    print("(1/dt, R, N, min, max mu, sigma)",[1/dt, R, N, np.min(d), np.max(d), np.mean(d),np.std(d,ddof=1)])
    ax.set_xlim([-0.5, 0.5]) # pyright: ignore[reportArgumentType]
    kde = st.gaussian_kde(d)
    x0 = np.linspace(np.min(d), np.max(d), num=1000)
    k1 = kde(x0)
    ax.plot(x0, k1, color = 'black')
    plt.hist(d, bins=200, density=True, color = 'orange')
    plt.show()

# ------------------------ 7. MJD Set Jumps Simulation -------------------------
# ------------------------------ 7.1 MJD Path Sim ------------------------------

def JumpTime_setJ(j0, lam):
    '''
    Jump Times - generates N jump times in [0,1] with lam param
    '''
    JT0 = np.zeros(j0+1)
    for j in range(0,j0+1):
        JT0[j] = np.random.default_rng().exponential(lam)
    s0 = np.cumsum(JT0)
    s1 = s0/s0[j0]
    s2 = np.delete(s1,j0)
    return s2

def countProc_setJ(j0, lam, dt):
    '''
    Poisson Counting Process
    '''
    jt = JumpTime_setJ(j0, lam)
    count = np.zeros(int(1 / dt))
    for t in range(1, int(1 / dt)):
        a = np.where(jt > (t * dt), 0, jt)
        b = np.max(a)
        if b == 0:
            count[t] = 0
        else:
            count[t] = list(jt).index(b) + 1
    return count
    
def JumpAmp_setJ(j0, lam, alpha, delta):
    '''
    Jump Amplitude
    '''
    JA = np.zeros(j0)
    for j in range(j0):
        JA[j] = np.random.default_rng().lognormal(alpha, delta) - 1
    return JA

def PoisProc_setJ(lam, alpha, delta, dt, j0):
    '''
    Poisson Process
    '''
    J = np.zeros(int(1 / dt))
    if j0 == 0:
        return J
    else:
        count_t = countProc_setJ(j0, lam, dt)
        y_j = JumpAmp_setJ(j0, lam, alpha, delta)
        for t in range(int(1 / dt)):
            if count_t[t] == 0:
                J[t] = 0
            else:
                J[t] = np.sum(y_j[0:(int(count_t[t] - 1))])
        return J

def MJD_setJ(mu, sigma, S0, lam, alpha, delta, dt, j0):
    '''
    MJD - Levy Process; sums GBM and Poisson Point Process
    '''
    path1 = mc.gbmP0(mu, sigma, S0, 1, dt)
    logS = np.log(path1)
    J1 = PoisProc_setJ(lam, alpha, delta, dt, j0)
    logS1 = np.add(logS,J1)
    S1 = np.exp(logS1)
    return S1

def LevyPlot_setJ(mu, sigma, S0, lam, alpha, delta, dt, j0):
    '''
    Plot Levy Process
    '''
    S1 = MJD_setJ(mu, sigma, S0, lam, alpha, delta, dt, j0)
    fig, ax = plt.subplots(figsize=[11, 5])
    ax.plot(np.arange(int(1 / dt)), S1, '-o', label='$S_t$', ms=1, alpha=0.6)
    plt.show()

# ------------------------------ 7.2 MJD Simulation ----------------------------
def MJDsim_setJ(r, mu, sigma, lam, alpha, delta, S0, dt, K, n, j0):
    spot = MJD_setJ(mu, sigma, S0, lam, alpha, delta, dt, j0)
    call_Jump = np.zeros(int(1 / dt))
    delta_Jump = np.zeros(int(1 / dt))
    for t in range(0, int(1 / dt)):
        call_Jump[t] = Call_Jump(sigma, spot[t],1,((t+1) / int(1 / dt)),K,r, lam, alpha, delta, n)
        delta_Jump[t] = Delta_MJD(spot[t],K,r,1,((t+1) / int(1 / dt)),sigma,lam,alpha,delta,n)
    return [spot,call_Jump,delta_Jump]

def MJDplot_setJ(r, mu, sigma, S0, dt, K, lam, alpha, delta, n, j0):
    '''
    Time series of MJD option price (Eur Call) - Returns spot, call price & delta
    '''
    x0 = MJDsim_setJ(r, mu, sigma, lam, alpha, delta, S0, dt, K, n, j0)
    fig, ax = plt.subplots(figsize=[11, 5])
    ax.plot(np.arange(int(1 / dt)), x0[0], '-o', label='$S_t$', ms=1, alpha=0.6)
    ax.plot(np.arange(int(1 / dt)), x0[1], '-o', label='Call$_t$', ms=1, alpha=0.6)
    # ax.plot(np.arange(int(1 / dt)), x0[2], '-o', label='Call$_t$', ms=1, alpha=0.6)
    plt.show()
    
# ---------------------------- 7.3 MJD Trading Sim -----------------------------
def dhedgeMJD_setJ(r, mu, sigma, lam, alpha, delta, S0, dt, K, n, R, j0):
    x0 = MJDsim_setJ(r, mu, sigma, lam, alpha, delta, S0, dt, K, n, j0)
    S = x0[0]
    V = x0[1]
    delta = x0[2]
    reb = int(1/(dt * R))
    qS = np.zeros(int(1 / dt))
    qB = np.zeros(int(1 / dt))
    prof = np.zeros(int(1 / dt))
    prof_adj = np.zeros(int(1 / dt))
    valS = np.zeros(int(1 / dt))
    valB = np.zeros(int(1 / dt))
    B = np.zeros(int(1 / dt))
    B[0] = 1
    qS[0] = delta[0]
    qB[0] = V[0] - delta[0] * S[0]
    valS[0] = qS[0] * S[0]
    valB[0] = qB[0] * B[0]
    prof[0] = valS[0] + valB[0] - V[0]
    for t in range(1,int(1 / dt)):
        if t in range(reb,int(1 / dt),reb):
            B[t] = B[t-1] * np.exp(r * dt)
            qS[t] = delta[t]
            qB[t] = qB[t-1] - S[t] * (delta[t] - delta[t-reb]) / B[t]
            valS[t] = qS[t] * S[t]
            valB[t] = qB[t] * B[t]
            prof[t] = valS[t] + valB[t] - V[t]
            prof_adj[t] = np.exp(-r * dt * t) * prof[t] / V[0]
        else:
            qS[t] = qS[t-1]
            qB[t] = qB[t-1]
            valS[t] = qS[t] * S[t]
            valB[t] = qB[t]
            prof[t] = valS[t] + valB[t] - V[t]
            prof_adj[t] = np.exp(-r * dt * t) * prof[t] / V[0]
    return [prof,prof_adj,valS,valB,qS,qB,B,S,V,delta]

def dhedgeMJDplot_setJ(r, mu, sigma, lam, alpha, delta, S0, dt, K, n, R, j0):
    x0 = dhedgeMJD_setJ(r, mu, sigma, lam, alpha, delta, S0, dt, K, n, R, j0)
    for j in range(9):
        fig, ax = plt.subplots(figsize=[11, 5])
        ax.plot(np.arange(int(1 / dt)), x0[j], '-o', label='$S_t$', ms=1, alpha=0.6)
        plt.show()

# ------------------------------------ END -------------------------------------