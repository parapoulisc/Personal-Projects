import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import multiprocessing as mp
import time

import Option as op

# ------------------------ 1. Diffusion Simulations ------------------------
def BIncr(deltaT):
    '''
    Generates Brownian Increment variance deltaT
    '''
    return np.random.default_rng().normal(0,deltaT ** 0.5)

def Bpath(deltaT):
    '''
    Simulates approximation to Wiener Process on unit interval with time interval deltaT
    '''
    path = np.zeros(int(1 / deltaT))
    fig, ax = plt.subplots(figsize=[11, 5])
    for t in range(1, int(1 / deltaT)):
        path[t] = path[t-1] + BIncr(deltaT)
    ax.plot(np.arange(int(1 / deltaT)), path, '-o', label='$W_t$', ms=1, alpha=0.6)
    plt.show()

def GBM_EM(mu, sigma, S0, deltaT):
    '''
    Generate GBM (Euler-Maruyama method) under P
    Not robust - r=0
    '''
    path = np.zeros(int(1 / deltaT))
    path[0] = S0
    for t in range(1, int(1 / deltaT)):
        path[t] = path[t-1] + mu * path[t-1] * deltaT + sigma * path[t-1] * BIncr(deltaT)
    return path

def GBM_M(mu, sigma, S0, deltaT):
    '''
    Generate GBM (Milstein method) under P
    Not robust - r=0
    '''
    path = np.zeros(int(1 / deltaT))
    path[0] = S0
    for t in range(1, int(1 / deltaT)):
        deltaW = BIncr(deltaT)
        path[t] = path[t-1] + mu * path[t-1] * deltaT + sigma * path[t-1] * deltaW + 0.5 * path[t-1] * sigma ** 2 * (deltaW ** 2 - deltaT)
    return path

def GBMplot(mu, sigma, S0, deltaT, N):
    '''
    Plot multiple N independent GBMs by Milstein method under P
    '''
    fig, ax = plt.subplots(figsize=[11, 5])
    GBMpaths = np.zeros((N, int(1 / deltaT)))
    for j in range(N):
        GBMpaths[j] = GBM_M(mu, sigma, S0, deltaT)
        ax.plot(np.arange(int(1 / deltaT)), GBMpaths[j], '-o', label='$S_t$', ms=1, alpha=0.6)
    plt.show()

def GBM_EM_Q(r, sigma, S0, deltaT):
    '''
    Generate GBM under Q by Euler-Maruyama Approxixmation
    '''
    path = np.zeros(int(1 / deltaT))
    path[0] = S0
    for t in range(1, int(1 / deltaT)):
        deltaW = BIncr(deltaT)
        path[t] = np.exp(np.log(path[t-1]) + (r - 0.5 * sigma ** 2) * deltaT + sigma * deltaW)
    return path[int(1 / deltaT - 1)]

def GBM_Mil_Q(r, sigma, S0, deltaT):
    '''
    Generate GBM under Q by Milstein Approximiation
    '''
    path = np.zeros(int(1 / deltaT))
    path[0] = np.log(S0)
    for t in range(1, int(1 / deltaT)):
        deltaW = BIncr(deltaT)
        path[t] = path[t-1] + (r - 0.5 * sigma ** 2) * deltaT + sigma * deltaW + 0.5 * sigma * (deltaW ** 2 - deltaT)
    return np.exp(path[int(1 / deltaT) - 1])

# ------------------------ 2. Monte Carlo Functions (I) ------------------------
def MC_EuroCall_EM(r, sigma, S0, deltaT, K, n):
    '''
    Monte Carlo Simulation for european call by Euler-Maruyama.
    
    Version 1.0.
    '''
    sum = np.zeros(n)
    for j in range(n):
        sum[j] = sum[j-1] + np.max([GBM_EM_Q(r, sigma, S0, deltaT) - K, 0])
    return sum

def MC_EuroCall_M(r, sigma, S0, deltaT, K, n):
    '''
    Monte Carlo Simulation for european call by Milstein.
    
    Version 1.1
    '''
    sum = np.zeros(n)
    for j in range(n):
        sum[j] = sum[j-1] + np.max([GBM_Mil_Q(r, sigma, S0, deltaT) - K, 0])
    return sum

# ------------------------ 3. Monte Carlo Convergence ------------------------
def Conv(r, sigma, S0, deltaT, K, n, p):
    '''
    Evaluate convergence of MC algo
    '''
    x0 = MC_EuroCall_M(r, sigma, S0, deltaT, K, n)
    x1 = np.zeros(n)
    for j in range(n):
        x1[j] = np.absolute(np.exp(-r)*(x0[j])/(j + 1) - p)
        print("Iteration",j,":", x1[j])
    fig, ax = plt.subplots(figsize=[11, 5])
    ax.loglog(np.arange(n), x1, '-o', label='$S_t$', ms=1, alpha=0.6)
    plt.show()
    
def Conv0(r, sigma, S0, deltaT, K, n, k):
    '''
    Convergence of multiple MC paths
    '''
    fig, ax = plt.subplots(figsize=[11, 5])
    MCpaths = np.zeros((k, n))
    MCpaths1 = np.zeros((k, n))
    for i in range(k):
        MCpaths[i] = MC_EuroCall_EM(r, sigma, S0, deltaT, K, n)
    for i in range(k):
        for j in range(n):
            MCpaths1[i][j] = MCpaths[i][j]/(j + 1) - 19.386
        ax.plot(np.arange(n), MCpaths1[i], '-o', label='$S_t$', ms=1, alpha=0.6)
        ax.set_ylim([-10, 10]) # pyright: ignore[reportArgumentType]
    plt.show()

def var(r, sigma, S0, deltaT, K, n, p, k):
    '''
    Evaluate asymptotic properties of MC estimators; k independent MC sims
    '''
    s1 = np.zeros(k)
    s2 = np.zeros(k)
    for j in range(k):
        s1[j] = s1[j-1] + (MC_EuroCall_M(r, sigma, S0, deltaT, K, n)[int(n - 1)]/n - p) ** 2
    for i in range(k):
        s2[i] = ( s1[i] / (i + 1) ) ** 0.5
    fig, ax = plt.subplots(figsize=[11, 5])
    ax.plot(np.arange(k), s2, '-o', label='$S_t$', ms=1, alpha=0.6)
    plt.show()
    
def hist(r, sigma, S0, deltaT, K, n, p, k):
    '''
    Histogram of point estimate of k independent MC sims
    '''
    fig, ax = plt.subplots(figsize=[11, 5])
    h0 = np.zeros(k)
    for j in range(k):
        h0[j] = (MC_EuroCall_M(r, sigma, S0, deltaT, K, n)[int(n - 1)]/n - p) / p
    # plt.hist(h0, bins=100)
    # plt.show()
    print(n, deltaT, np.mean(h0),np.var(h0)**0.5)

# ----------------------- 3. Monte Carlo Functions (II) ------------------------
def MC_EuroCall_EM_F(r, sigma, S0, deltaT, K, n):
    '''
    Monte Carlo (EM).
    
    Version 2.0
    '''
    sum = np.zeros(n)
    for j in range(1,n):
        sum[j] = (sum[j-1] * (j - 1) + np.max([GBM_EM_Q(r, sigma, S0, deltaT) - K, 0])) / j
    return np.exp(-r) * sum[n-1]

# ----------------------------------- BREAK ------------------------------------
# ------------------- 4. Accelerated Diffusion Sims (III) ----------------------
def dW(T, deltaT):
    '''
    Generates Brownian Increment, variance deltaT
    '''
    return np.random.normal(0, deltaT ** 0.5, size=int(T / deltaT))

def Bpath_f(T, deltaT):
    '''
    Generates Brownian Path
    '''
    return np.cumsum(dW(T, deltaT))

def gbmP0(mu, sigma, S0, T, deltaT):
    '''
    Generates GBM sim under P
    '''
    dw = dW(T, deltaT)
    path = np.zeros(int(T / deltaT) + 1)
    path[0] = S0
    for t in range(1, int(T / deltaT) + 1):
        path[t] = path[t-1] * np.exp((mu - 0.5 * sigma ** 2) * deltaT + sigma * dw[t-1])
    return path

def gbmQ0(r, sigma, S0, T, deltaT):
    '''
    Generates GBM sim under Q
    '''
    dw = dW(T, deltaT)
    path = np.zeros(int(T / deltaT))
    path[0] = S0
    for t in range(1, int(T / deltaT)):
        path[t] = path[t-1] * np.exp((r - 0.5 * sigma ** 2) * deltaT + sigma * dw[t-1])
    return path

def plot(mu, sigma, S0, T, deltaT):
    '''
    Plot GBM sample path
    '''
    path = gbmP0(mu, sigma, S0, T, deltaT)
    fig, ax = plt.subplots(figsize=[11, 5])
    ax.plot(np.arange(int(T / deltaT)), path, '-o', ms=0.1, alpha=0.6)
    plt.show()

# ---------------------- 5. Monte Carlo Functions (III) ------------------------
def MC(r, sigma, S0, T, deltaT, K, n):
    '''
    Monte Carlo European Option
    
    Version 3.0
    '''
    theo = op.Option(S0, K, r, T, 0, sigma, 'European', 'call', deltaT)
    p_mc = np.zeros(n)
    z = np.zeros(n)
    for j in range(n):
        p_mc[j] = gbmQ0(r, sigma, S0, T, deltaT)[int(1 / deltaT) - 1]
    a_MC = np.exp(-r * T) * np.sum(np.maximum((p_mc - K),z)) / n
    a_BS = theo.price()
    eps = abs(a_BS/a_MC-1)
    return ['BS', a_BS, 'MC', a_MC, 'eps', eps]

# ------------------- 6. Accelerated Diffusion Sims (IV) -----------------------
def dW0(T, deltaT, N):
    '''
    Draws N white noise increments with deltaT time step for T time.
    
    Returns matrix (array) with N rows and 1/deltaT columns.
    '''
    return np.random.normal(0, deltaT ** 0.5, size=(N, int(T / deltaT)))

def gbmP1(mu, sigma, S0, T, deltaT, N):
    """Draws N GBM's jointly under P.

    Args:
        mu (_type_): _description_
        
        sigma (_type_): _description_
        
        S0 (_type_): _description_
        
        T (_type_): _description_
        
        deltaT (_type_): _description_
        
        N (_type_): _description_

    Returns:
        np.array: Returns path, an (Nx1/deltaT) matrix with sample paths.
    """
    dw = dW0(T, deltaT, N)
    path = np.zeros((N, int(T / deltaT) + 1))
    path[:,0] = S0
    for t in range(1, int(T / deltaT) + 1):
        path[:,t] = path[:,t-1] * np.exp((mu - 0.5 * sigma ** 2) * deltaT + sigma * dw[:,t-1])
    return path

def gbmQ1(r, sigma, S0, T, deltaT, N):
    '''
    Draws N GBM's for Monte Carlo jointly under Q.
    
    Returns path, an (Nx1/deltaT) matrix with sample paths.
    '''
    dw = dW0(T, deltaT, N)
    path = np.zeros((N, int(T / deltaT) + 1))
    path[:,0] = S0
    for t in range(1, int(T / deltaT) + 1):
        path[:,t] = path[:,t-1] * np.exp((r - 0.5 * sigma ** 2) * deltaT + sigma * dw[:,t-1])
    return path

# ---------------------- 7. Monte Carlo Functions (IV) -------------------------
def MC0(r, sigma, S0, T, deltaT, K, N):
    '''
    Monte Carlo European Option
    
    Version 4.0
    '''
    theo = op.Option(S0, K, r, T, 0, sigma, 'European', 'call', deltaT)
    S_T = gbmQ1(r , sigma, S0, T, deltaT, N)[:,int(1 / deltaT)]
    a_MC = np.exp(-r * T) * np.sum(np.maximum((S_T - K), 0.0)) / N
    a_BS = theo.price()
    eps = abs(a_BS/a_MC-1)
    return ['BS', a_BS.round(5), 'MC', a_MC.round(5), 'eps', eps.round(5)]

# --------------------- 8. Accelerated Diffusion Sims (V) ----------------------
def dW0F(T, deltaT, N):
    return (deltaT ** 0.5) * np.random.standard_normal(size=(N, int(T / deltaT)))

def gbmP2(mu, sigma, S0, T, deltaT, N):
    dw = dW0F(T, deltaT, N)
    path = np.zeros((N, int(T / deltaT) + 1))
    path[:,0] = S0
    for t in range(1, int(T / deltaT) + 1):
        path[:,t] = path[:,t-1] * np.exp((mu - 0.5 * sigma ** 2) * deltaT + sigma * dw[:,t-1])
    return path

def gbmP3(mu, sigma, S0, T, deltaT, N):
    dw = dW0F(T, deltaT, N)
    increments = (mu - 0.5 * sigma ** 2) * deltaT + sigma * dw
    log_paths = np.concatenate([np.zeros((N, 1)), np.cumsum(increments, axis = 1)], axis = 1)
    path = S0 * np.exp(log_paths)
    return path

# ------------------- 9. Multiproccesing Diffusion Sims (VI) -------------------
def gbm_chunk(args):
    mu, sigma, S0, T, deltaT, n_paths, seed = args
    rng = np.random.default_rng(seed)
    steps = int(T / deltaT)
    dw = rng.standard_normal(size=(steps, n_paths)) * np.sqrt(deltaT)
    increments = (mu - 0.5 * sigma**2) * deltaT + sigma * dw
    log_paths = np.vstack([np.zeros(n_paths), np.cumsum(increments, axis=0)])
    return S0 * np.exp(log_paths)

def gbmP4(mu, sigma, S0, T, deltaT, N):
    # Use all available CPU cores
    n_jobs = mp.cpu_count()
    
    base = N // n_jobs
    remainder = N % n_jobs
    chunk_sizes = [base] * n_jobs
    for i in range(remainder):
        chunk_sizes[i] += 1  # distribute remainder

    # Prepare arguments for each process
    seeds = np.random.SeedSequence().spawn(n_jobs)
    args_list = [(mu, sigma, S0, T, deltaT, chunk_sizes[i], seeds[i].entropy) for i in range(n_jobs)]
    
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(gbm_chunk, args_list)
    
    # Concatenate all chunks
    return np.hstack(results)

# --------------------------- 9. MJD Diffusion Sims ----------------------------
def JumpTime(N, lambda0):
    """
    Generate N jump times in the interval [0,1] for a Poisson process with intensity lambda0.

    Parameters
    ----------
    N : int
        Number of jumps to simulate.
    lambda0 : float
        Poisson process rate parameter (mean inter-arrival time).

    Returns
    -------
    np.ndarray
        Array of N normalized jump times in [0, 1).
    """
    if N == 0:
        return np.array([])
    inter_arrivals = np.random.default_rng().exponential(lambda0, size = N + 1)
    jump_times = np.cumsum(inter_arrivals)
    jump_times = jump_times / jump_times[N]
    jump_times = np.delete(jump_times, N)
    return jump_times

def countProc(N, lambda0, deltaT):
    """
    Simulate a Poisson counting process over [0,1] with given jump times.

    Parameters
    ----------
    N : int
        Number of jumps to simulate.
    lambda0 : float
        Poisson process rate parameter (mean inter-arrival time).
    deltaT : float
        Time step size (fraction of year).

    Returns
    -------
    np.ndarray
        Array of counts at each time step, representing the cumulative number of jumps up to each time.
    """
    steps = int(1 / deltaT) + 1
    jt = JumpTime(N, lambda0)
    t_grid = np.linspace(0, 1, steps)
    if N == 0 or jt.size == 0:
        return np.zeros(steps, dtype=int)
    counts = np.searchsorted(jt, t_grid, side='right')
    return counts

def JumpAmp(N, lambda0, alpha, delta):
    """
    Generate amplitudes for N jumps using a lognormal distribution.

    Parameters
    ----------
    N : int
        Number of jumps.
    lambda0 : float
        Poisson process rate parameter (unused in this function, included for interface compatibility).
    alpha : float
        Mean of the logarithm of jump size.
    delta : float
        Standard deviation of the logarithm of jump size.

    Returns
    -------
    np.ndarray
        Array of N jump amplitudes (lognormal random variables minus 1).
    """
    if N == 0:
        return np.zeros(0)
    return np.random.default_rng().lognormal(alpha, delta, size=N) - 1

def PoisProc(lambda0, alpha, delta, deltaT):
    """
    Simulate a compound Poisson process over [0,1] with lognormal jump amplitudes.

    Parameters
    ----------
    lambda0 : float
        Poisson process rate parameter (expected number of jumps).
    alpha : float
        Mean of the logarithm of jump size.
    delta : float
        Standard deviation of the logarithm of jump size.
    deltaT : float
        Time step size (fraction of year).

    Returns
    -------
    np.ndarray
        Array of process values at each time step (cumulative sum of jump amplitudes).
        Length is N+1, where N = int(1/deltaT): first element is 0, others are cumulative jump sums.
    """
    Nsteps = int(1 / deltaT) + 1
    N_jumps = np.random.default_rng().poisson(lambda0)
    if N_jumps == 0:
        return np.zeros(Nsteps)
    jump_times = JumpTime(N_jumps, lambda0)
    jump_amps = JumpAmp(N_jumps, lambda0, alpha, delta)
    t_grid = np.linspace(0, 1, Nsteps)
    # For each time, sum all jump amplitudes whose jump time <= t
    # Use searchsorted for vectorized cumulative sum
    # Get indices of rightmost jump included at each time
    jump_idx = np.searchsorted(jump_times, t_grid, side='right')
    # Cumulative sum of jump amplitudes for all jumps
    cumsum_amps = np.concatenate([[0], np.cumsum(jump_amps)])
    # For each time, process value is cumsum_amps[jump_idx[t]]
    J = cumsum_amps[jump_idx]
    return J

def MJD(mu, sigma, S0, lambda0, alpha, delta, deltaT):
    """
    Simulate a Merton Jump Diffusion (MJD) process by combining a geometric Brownian motion (GBM)
    with a compound Poisson jump process.

    Parameters
    ----------
    mu : float
        Drift of the GBM component.
    sigma : float
        Volatility of the GBM component.
    S0 : float
        Initial asset price.
    lambda0 : float
        Poisson process rate parameter (expected number of jumps).
    alpha : float
        Mean of the logarithm of jump size.
    delta : float
        Standard deviation of the logarithm of jump size.
    deltaT : float
        Time step size (fraction of year).

    Returns
    -------
    np.ndarray
        Simulated asset price path under Merton Jump Diffusion model.
        Length is N+1, with S1[0] = S0 and S1[1:] subsequent prices.
    """
    Nsteps = int(1 / deltaT) + 1
    path_gbm = gbmP0(mu, sigma, S0, 1, deltaT)  # length Nsteps
    logS = np.log(path_gbm)
    J1 = PoisProc(lambda0, alpha, delta, deltaT)
    logS1 = logS + J1
    S1 = np.exp(logS1)
    return S1

def LevyPlot(mu, sigma, S0, lambda0, alpha, delta, deltaT):
    '''
    Plot Levy Process
    '''
    S1 = MJD(mu, sigma, S0, lambda0, alpha, delta, deltaT)
    fig, ax = plt.subplots(figsize=[11, 5])
    ax.plot(np.arange(int(1 / deltaT) + 1), S1, '-o', label='$S_t$', ms=1, alpha=0.6)
    plt.show()
# ------------------------------ END ------------------------------