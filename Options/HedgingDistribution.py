import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import math as m

import Option as op
import MonteCarlo as mc

# ------------------------ 1. Black-Scholes Simulation ------------------------
def BSsim(r, mu, sigma, S0, K, type, deltaT):
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
    deltaT : float
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
    N = int(1 / deltaT)
    t = [j / N for j in range(N + 1)]
    spot = mc.gbmP0(mu, sigma, S0, 1, deltaT)
    underlying = op.Spot(spot,r, t, 1, deltaT)
    option = op.OptionEng(spot, K, r, 1, t, sigma, 'European', type, op.BlackScholesEngine(), deltaT)
    scal_vol = underlying.VarReal()
    return t, spot, option.price(), option.delta(), option.gamma(), option.dollarGamma(), scal_vol, option.theta()

def BSplot(r, mu, sigma, S0, K, type, deltaT):
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
    deltaT : float
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
    t, spot, vBS, deltaBS, gammaBS, dolgammaBS, spotVol, theta = BSsim(r, mu, sigma, S0, K, type, deltaT)

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
      
# --------------------- 2. Delta Hedging in BS Simulation ---------------------
def dhedge(r, mu, sigma, S0, K, type, deltaT, R):
    """
    Simulates a discrete-time delta-hedging strategy under the Black-Scholes model with periodic rebalancing of the hedge portfolio.

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
    deltaT : float
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
    time, S, V, delta, gamma, dolGamma, vol, theta = BSsim(r, mu, sigma, S0, K, type, deltaT)
    
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
            B[t] = B[t-1] * np.exp(r * deltaT)
            qS[t] = delta[t]
            qB[t] = qB[t-1] - S[t] * (delta[t] - delta[t-reb]) / B[t]
            vS[t] = qS[t] * S[t]
            vB[t] = qB[t] * B[t]
            prof[t] = vS[t] + vB[t] - V[t]
            prof_adj[t] = np.exp(-r * deltaT * t) * prof[t] / V[0]
        else:
            B[t] = B[t-1] * np.exp(r * deltaT)
            qS[t] = qS[t-1]
            qB[t] = qB[t-1]
            vS[t] = qS[t] * S[t]
            vB[t] = qB[t] * B[t] 
            prof[t] = vS[t] + vB[t] - V[t]
            prof_adj[t] = np.exp(-r * deltaT * t) * prof[t] / V[0]
    return prof, prof_adj, vS, vB, qS, qB, B, S, V, delta, gamma, dolGamma, vol, theta, time

def dhedgePlot(r, mu, sigma, S0, K, type, deltaT, R):
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
    deltaT : float
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
    prof, prof_adj, valS, valB, qS, qB, B, S, V, delta, gamma, dolGamma, vol, theta, time = dhedgeV(r, mu, sigma, S0, K, type, deltaT, R)
    labels = ['Nominal Profit', 'Net Return', 'Value of Stock Portfolio',       'Value of Bond Portfolio', 'Stock Holdings','Bond Holdings', 'Bond Price', 'Stock Price', 'Option Price', 'Gamma', 'Dollar Gamma', 'Realised Variance (Ann)','GammaRent','Theta','Gamma Rent, Theta','Gamma Rent less Theta','-cumGamRentLessTheta','PDE Gamma Term','PDE int term','PDE Sum']
    fig, axes = plt.subplots(5, 4, figsize=(18, 12))
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
    
def dhedgeHist(r, mu, sigma, S0, deltaT, K, R, N):
    """
    Simulates a delta-hedging strategy multiple times and plots a histogram 
    of the final adjusted portfolio profits.
    
    Delta-hedging simulated by dhedgeVF() to optimise run time.

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
    deltaT : float
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
        p[j] = dhedgeVF(r,mu,sigma,S0,deltaT,K,R)[1][int(1/deltaT)]
    plt.hist(p, bins=500)
    plt.show()
    
def dhedgeError(r, mu, sigma, S0, deltaT, K, R, N):
    """
    Simulates a delta-hedging strategy multiple times and returns the final
    adjusted portfolio profits for each simulation.
    
    Delta-hedging simulated by dhedgeVF() to optimise run time.

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
    deltaT : float
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
        p[j] = dhedgeVF(r,mu,sigma,S0,deltaT,K,R)[1][int(1/deltaT)]
    return p

def dhedgeHistPDF(r, mu, sigma, S0, deltaT, K, R, N):
    """
    Simulates a delta-hedging strategy multiple times and plots a histogram
    and kernel density estimate (KDE) of the final adjusted portfolio profits.
    
    Delta-hedging simulated by dhedgeVF() to optimise run time.

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
    deltaT : float
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
    d = dhedgeError(r,mu,sigma,S0,deltaT,K,R,N)
    print("(r, mu, sigma, S0, 1/deltaT, K, R, N, min, max, mu_eps, sigma_eps, kurtosis_eps, skew_eps)",[r, mu, sigma, S0, 1/deltaT, K, R, N, np.min(d), np.max(d), np.mean(d),np.std(d,ddof=1), st.kurtosis(d),st.skew(d)])
    fig, ax = plt.subplots(figsize=[11, 5])
    ax.set_xlim([np.min(d), np.max(d)]) # pyright: ignore[reportArgumentType]
    kde = st.gaussian_kde(d)
    x0 = np.linspace(np.min(d), np.max(d), num=1000)
    k1 = kde(x0)
    ax.plot(x0, k1, color = 'black')
    plt.hist(d, bins=200, density=True, color = 'orange')
    plt.show()

#------------------ 3. Opt Delta Hedging Functions (path N=1) ------------------
def dhedgeV(r, mu, sigma, S0, K, type, deltaT, R):
    """
    Vectorized delta-hedging under Black-Scholes.
    
    Fasted diffusion path algo but not optimised for overall speed. Returns numerous Greeks for plotting function.
    """
    time, S, V, delta, gamma, dolGamma, vol, theta = BSsim(r, mu, sigma, S0, K, type, deltaT)
    N = len(time)
    reb = int(N / R)
    idx = np.arange(N) #effective time vector

    B = np.exp(r * deltaT * idx)

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

    valS = qS * S
    valB = qB * B
    prof = valS + valB - V
    prof_adj = np.exp(-r * deltaT * idx) * prof / V[0]

    return prof, prof_adj, valS, valB, qS, qB, B, S, V, delta, gamma, dolGamma, vol, theta, time

def BSsimF0(r, mu, sigma, S0, K, type, deltaT):
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
    deltaT : float
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
    N = int(1 / deltaT)
    t = [j / N for j in range(N + 1)]
    spot = mc.gbmP0(mu, sigma, S0, 1, deltaT)
    option = op.OptionEng(spot.T, K, r, 1, t, sigma, 'European', type, op.BlackScholesEngine(), deltaT)
    return t, spot.T, option.price(), option.delta()

def dhedgeVF(r, mu, sigma, S0, K, type, deltaT, R):
    """
    Vectorized delta-hedging under Black-Scholes.
    
    Optimised for speed rather than visualisation.
    """
    time, S, V, delta = BSsimF0(r, mu, sigma, S0, K, type, deltaT)
    N = len(time)
    reb = int(N / R)

    B = np.exp(r * deltaT * np.arange(N))

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
    prof_adj = np.exp(-r * deltaT * idx) * prof / V[0]

    return prof, prof_adj, vS, vB, qS, qB, B, S, V, delta, time

#---------------- 4. Opt Delta Hedging Functions (path genN) ------------------
def BSsimF1(r, mu, sigma, S0, K, type, deltaT, paths):
    """
    Simulates a geometric Brownian motion stock price path and computes
    the corresponding Black-Scholes option metrics at each time step.
    
    Jointly samples paths, collecting into matrix with paths (rows) x 1/deltaT (cols). Then collects option Greeks into same structure. 
    
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
    deltaT : float
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
    N = int(1 / deltaT)
    t = [j / N for j in range(N + 1)]
    spot = mc.gbmP2(mu, sigma, S0, 1, deltaT, paths)
    option = op.Option(spot, K, r, 1, t, sigma, 'European', type, deltaT)
    return t, spot, option.price(), option.delta()

def dhedgeVF1(r, mu, sigma, S0, K, type, deltaT, R, paths):
    """
    Vectorized delta-hedging under Black-Scholes.
    
    Handles jointly sampled paths.
    
    Efficient when drawing large numbers of paths for Monte Carlo.
    """
    time, S, V, delta = BSsimF1(r, mu, sigma, S0, K, type, deltaT, paths)
    N = len(time)
    reb = int(N / R)
    idx = np.arange(N) #effective time vector for bond
    tr_idx = np.vstack([np.arange(N) for i in range(paths)])

    B = np.exp(r * deltaT * idx)

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

    # cash required for stock trades at each step, discounted by bond price
    cash_flow = (S * delta_qS) / B[None, :]

    # cumulative financing from trades
    qB = qB[:, [0]] - np.cumsum(cash_flow, axis=1)

    vS = qS * S
    vB = qB * B
    prof = vS + vB - V
    prof_adj = np.exp(-r * deltaT * idx) * prof / V[:,0][0]

    return prof, prof_adj, vS, vB, qS, qB, B, S, V, delta, time

# ----------------------------- 5. MJD Simulation ------------------------------
def MJDsim(r, mu, sigma, lambda0, alpha, deltaJ, S0, deltaT, K, n, type):
    '''
    MJD option and delta simulation.
    '''
    N = int(1 / deltaT)
    t = [j / N for j in range(N + 1)]
    spot = mc.MJD(mu, sigma, S0, lambda0, alpha, deltaJ, deltaT)
    option = op.OptionEng(spot, K, r, 1, t, sigma, 'European', type, op.MertonJumpDiffusionEngine(lambda0, alpha, deltaJ, n), deltaT)
    return [t, spot, option.price(), option.delta()]

def MJDplot(r, mu, sigma, S0, deltaT, K, lambda0, alpha, deltaJ, n, type):
    '''
    Panel plot of MJD option simulation mirroring BSplot style.
    Plots spot price, option price, and delta over time.
    '''
    t, spot, vMJD, deltaMJD = MJDsim(r, mu, sigma, lambda0, alpha, deltaJ, S0, deltaT, K, n, type)

    labels = [
        ("Spot Price", spot),
        ("Option Price", vMJD),
        ("Delta", deltaMJD)
    ]

    # Use a 2x2 panel for clarity and to allow for future expansion (e.g. gamma)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (label, data) in zip(axes, labels):
        ax.plot(t, data, '-o', ms=1, alpha=0.6, label=label)
        ax.set_title(label)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True)
        ax.legend()

    # Hide any unused subplot(s)
    for j in range(len(labels), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    
# ------------------------------ NOT CLEANED WIP -------------------------------
# -------------------------- 6. Delta Hedging in MJD ---------------------------
def dhedgeMJD(r, mu, sigma, lambda0, alpha, delta, S0, deltaT, K, n, R):
    x0 = MJDsim(r, mu, sigma, lambda0, alpha, delta, S0, deltaT, K, n)
    S = x0[0]
    V = x0[1]
    delta = x0[2]
    reb = int(1/(deltaT * R))
    qS = np.zeros(int(1 / deltaT))
    qB = np.zeros(int(1 / deltaT))
    prof = np.zeros(int(1 / deltaT))
    prof_adj = np.zeros(int(1 / deltaT))
    valS = np.zeros(int(1 / deltaT))
    valB = np.zeros(int(1 / deltaT))
    B = np.zeros(int(1 / deltaT))
    B[0] = 1
    qS[0] = delta[0]
    qB[0] = V[0] - delta[0] * S[0]
    valS[0] = qS[0] * S[0]
    valB[0] = qB[0] * B[0]
    prof[0] = valS[0] + valB[0] - V[0]
    for t in range(1,int(1 / deltaT)):
        if t in range(reb,int(1 / deltaT),reb):
            B[t] = B[t-1] * np.exp(r * deltaT)
            qS[t] = delta[t]
            qB[t] = qB[t-1] - S[t] * (delta[t] - delta[t-reb]) / B[t]
            valS[t] = qS[t] * S[t]
            valB[t] = qB[t] * B[t]
            prof[t] = valS[t] + valB[t] - V[t]
            prof_adj[t] = np.exp(-r * deltaT * t) * prof[t] / V[0]
        else:
            B[t] = B[t-1] * np.exp(r * deltaT)
            qS[t] = qS[t-1]
            qB[t] = qB[t-1]
            valS[t] = qS[t] * S[t]
            valB[t] = qB[t] * B[t]
            prof[t] = valS[t] + valB[t] - V[t]
            prof_adj[t] = np.exp(-r * deltaT * t) * prof[t] / V[0]
    return [prof,prof_adj,valS,valB,qS,qB,B,S,V,delta]

def dhedgeMJDplot(r, mu, sigma, lambda0, alpha, delta, S0, deltaT, K, n, R):
    x0 = dhedgeMJD(r, mu, sigma, lambda0, alpha, delta, S0, deltaT, K, n, R)
    fig, ax = plt.subplots(figsize=[11, 5])
    ax.plot(np.arange(int(1 / deltaT)), x0[1], '-o', label='$S_t$', ms=1, alpha=0.6)
    plt.show()
    
def dhedgeMJDhist(r, mu, sigma, lambda0, alpha, delta, S0, deltaT, K, n, N, R):
    fig, ax = plt.subplots(figsize=[11, 5])
    p = np.zeros(N)
    for j in range(N):
        p[j] = dhedgeMJD(r, mu, sigma, lambda0, alpha, delta, S0, deltaT, K, n, R)[1][int(1/deltaT)-1]
    plt.hist(p, bins=500)
    plt.show()
    
def dhedgeMJDerror(r, mu, sigma, lambda0, alpha, delta, S0, deltaT, K, n, R, N):
    p = np.zeros(N)
    for j in range(N):
        p[j] = dhedgeMJD(r, mu, sigma, lambda0, alpha, delta, S0, deltaT, K, n, R)[1][int(1/deltaT)-1]
    return p

def dhedgeMJDhistpdf(r, mu, sigma, lambda0, alpha, delta, S0, deltaT, K, n, R, N):
    d = dhedgeMJDerror(r, mu, sigma, lambda0, alpha, delta, S0, deltaT, K, n, R, N)
    fig, ax = plt.subplots(figsize=[11, 5])
    kde = st.gaussian_kde(d)
    x0 = np.linspace(np.min(d), np.max(d), num=1000)
    k1 = kde(x0)
    ax.plot(x0, k1)
    plt.hist(d, bins=500, density=True)
    # tikzplotlib.save("A.tex")
    plt.show()

def plotsJ(r,mu,sigma,S0,deltaT,K,R,N,lambda0,alpha,delta,n):
    '''
    Final Plots for Jumps.
    '''
    fig, ax = plt.subplots(figsize=[11, 5])
    d = dhedgeMJDerror(r, mu, sigma, lambda0, alpha, delta, S0, deltaT, K, n, R, N)
    print("(1/deltaT, R, N, min, max mu, sigma)",[1/deltaT, R, N, np.min(d), np.max(d), np.mean(d),np.std(d,ddof=1)])
    ax.set_xlim([-0.5, 0.5]) # pyright: ignore[reportArgumentType]
    kde = st.gaussian_kde(d)
    x0 = np.linspace(np.min(d), np.max(d), num=1000)
    k1 = kde(x0)
    ax.plot(x0, k1, color = 'black')
    plt.hist(d, bins=200, density=True, color = 'orange')
    plt.show()

# ------------------- 7. Levy Process Simulation (Set Jumps) -------------------
def JumpTime0(j0, lambda0):
    '''
    Jump Times - generates N jump times in [0,1] with lambda0 param
    '''
    JT0 = np.zeros(j0+1)
    for j in range(0,j0+1):
        JT0[j] = np.random.default_rng().exponential(lambda0)
    s0 = np.cumsum(JT0)
    s1 = s0/s0[j0]
    s2 = np.delete(s1,j0)
    return s2

def countProc0(j0, lambda0, deltaT):
    '''
    Poisson Counting Process
    '''
    jt = JumpTime0(j0, lambda0)
    count = np.zeros(int(1 / deltaT))
    for t in range(1, int(1 / deltaT)):
        a = np.where(jt > (t * deltaT), 0, jt)
        b = np.max(a)
        if b == 0:
            count[t] = 0
        else:
            count[t] = list(jt).index(b) + 1
    return count
    
def JumpAmp0(j0, lambda0, alpha, delta):
    '''
    Jump Amplitude
    '''
    JA = np.zeros(j0)
    for j in range(j0):
        JA[j] = np.random.default_rng().lognormal(alpha, delta) - 1
    return JA

def PoisProc0(lambda0, alpha, delta, deltaT, j0):
    '''
    Poisson Process
    '''
    J = np.zeros(int(1 / deltaT))
    if j0 == 0:
        return J
    else:
        count_t = countProc0(j0, lambda0, deltaT)
        y_j = JumpAmp0(j0, lambda0, alpha, delta)
        for t in range(int(1 / deltaT)):
            if count_t[t] == 0:
                J[t] = 0
            else:
                J[t] = np.sum(y_j[0:(int(count_t[t] - 1))])
        return J

def MJD0(mu, sigma, S0, lambda0, alpha, delta, deltaT, j0):
    '''
    MJD - Levy Process; sums GBM and Poisson Point Process
    '''
    path1 = mc.gbmP0(mu, sigma, S0, 1, deltaT)
    logS = np.log(path1)
    J1 = PoisProc0(lambda0, alpha, delta, deltaT, j0)
    logS1 = np.add(logS,J1)
    S1 = np.exp(logS1)
    return S1

def LevyPlot0(mu, sigma, S0, lambda0, alpha, delta, deltaT, j0):
    '''
    Plot Levy Process
    '''
    S1 = MJD0(mu, sigma, S0, lambda0, alpha, delta, deltaT, j0)
    fig, ax = plt.subplots(figsize=[11, 5])
    ax.plot(np.arange(int(1 / deltaT)), S1, '-o', label='$S_t$', ms=1, alpha=0.6)
    plt.show()

# ----------------------- 8. MJD Simulation (Set Jumps) -----------------------
def MJDsim0(r, mu, sigma, lambda0, alpha, delta, S0, deltaT, K, n, j0):
    spot = MJD0(mu, sigma, S0, lambda0, alpha, delta, deltaT, j0)
    call_Jump = np.zeros(int(1 / deltaT))
    delta_Jump = np.zeros(int(1 / deltaT))
    for t in range(0, int(1 / deltaT)):
        call_Jump[t] = Call_Jump(sigma, spot[t],1,((t+1) / int(1 / deltaT)),K,r, lambda0, alpha, delta, n)
        delta_Jump[t] = Delta_MJD(spot[t],K,r,1,((t+1) / int(1 / deltaT)),sigma,lambda0,alpha,delta,n)
    return [spot,call_Jump,delta_Jump]

def MJDplot0(r, mu, sigma, S0, deltaT, K, lambda0, alpha, delta, n, j0):
    '''
    Time series of MJD option price (Eur Call) - Returns spot, call price & delta
    '''
    x0 = MJDsim0(r, mu, sigma, lambda0, alpha, delta, S0, deltaT, K, n, j0)
    fig, ax = plt.subplots(figsize=[11, 5])
    ax.plot(np.arange(int(1 / deltaT)), x0[0], '-o', label='$S_t$', ms=1, alpha=0.6)
    ax.plot(np.arange(int(1 / deltaT)), x0[1], '-o', label='Call$_t$', ms=1, alpha=0.6)
    # ax.plot(np.arange(int(1 / deltaT)), x0[2], '-o', label='Call$_t$', ms=1, alpha=0.6)
    plt.show()
    
# -------------------- 9. Delta Hedging in MJD (Set Jumps) --------------------
def dhedgeMJD0(r, mu, sigma, lambda0, alpha, delta, S0, deltaT, K, n, R, j0):
    x0 = MJDsim0(r, mu, sigma, lambda0, alpha, delta, S0, deltaT, K, n, j0)
    S = x0[0]
    V = x0[1]
    delta = x0[2]
    reb = int(1/(deltaT * R))
    qS = np.zeros(int(1 / deltaT))
    qB = np.zeros(int(1 / deltaT))
    prof = np.zeros(int(1 / deltaT))
    prof_adj = np.zeros(int(1 / deltaT))
    valS = np.zeros(int(1 / deltaT))
    valB = np.zeros(int(1 / deltaT))
    B = np.zeros(int(1 / deltaT))
    B[0] = 1
    qS[0] = delta[0]
    qB[0] = V[0] - delta[0] * S[0]
    valS[0] = qS[0] * S[0]
    valB[0] = qB[0] * B[0]
    prof[0] = valS[0] + valB[0] - V[0]
    for t in range(1,int(1 / deltaT)):
        if t in range(reb,int(1 / deltaT),reb):
            B[t] = B[t-1] * np.exp(r * deltaT)
            qS[t] = delta[t]
            qB[t] = qB[t-1] - S[t] * (delta[t] - delta[t-reb]) / B[t]
            valS[t] = qS[t] * S[t]
            valB[t] = qB[t] * B[t]
            prof[t] = valS[t] + valB[t] - V[t]
            prof_adj[t] = np.exp(-r * deltaT * t) * prof[t] / V[0]
        else:
            qS[t] = qS[t-1]
            qB[t] = qB[t-1]
            valS[t] = qS[t] * S[t]
            valB[t] = qB[t]
            prof[t] = valS[t] + valB[t] - V[t]
            prof_adj[t] = np.exp(-r * deltaT * t) * prof[t] / V[0]
    return [prof,prof_adj,valS,valB,qS,qB,B,S,V,delta]

def dhedgeMJDplot0(r, mu, sigma, lambda0, alpha, delta, S0, deltaT, K, n, R, j0):
    x0 = dhedgeMJD0(r, mu, sigma, lambda0, alpha, delta, S0, deltaT, K, n, R, j0)
    for j in range(9):
        fig, ax = plt.subplots(figsize=[11, 5])
        ax.plot(np.arange(int(1 / deltaT)), x0[j], '-o', label='$S_t$', ms=1, alpha=0.6)
        plt.show()

# ------------------------------------ END -------------------------------------