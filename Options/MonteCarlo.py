import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import multiprocessing as mp

import Assets as ast
import Pricing as pr

# ------------------------ 1. Diffusion Simulations ------------------------
def BIncr(dt):
    '''
    Generates Brownian Increment variance dt
    '''
    return np.random.default_rng().normal(0,dt ** 0.5)

def Bpath(dt):
    '''
    Simulates approximation to Wiener Process on unit interval with time interval dt
    '''
    path = np.zeros(int(1 / dt))
    fig, ax = plt.subplots(figsize=[11, 5])
    for t in range(1, int(1 / dt)):
        path[t] = path[t-1] + BIncr(dt)
    ax.plot(np.arange(int(1 / dt)), path, '-o', label='$W_t$', ms=1, alpha=0.6)
    plt.show()

def GBM_EM(mu, sigma, S0, dt):
    '''
    Generate GBM (Euler-Maruyama method) under P
    Not robust - r=0
    '''
    path = np.zeros(int(1 / dt))
    path[0] = S0
    for t in range(1, int(1 / dt)):
        path[t] = path[t-1] + mu * path[t-1] * dt + sigma * path[t-1] * BIncr(dt)
    return path

def GBM_M(mu, sigma, S0, dt):
    '''
    Generate GBM (Milstein method) under P
    Not robust - r=0
    '''
    path = np.zeros(int(1 / dt))
    path[0] = S0
    for t in range(1, int(1 / dt)):
        deltaW = BIncr(dt)
        path[t] = path[t-1] + mu * path[t-1] * dt + sigma * path[t-1] * deltaW + 0.5 * path[t-1] * sigma ** 2 * (deltaW ** 2 - dt)
    return path

def GBMplot(mu, sigma, S0, dt, N):
    '''
    Plot multiple N independent GBMs by Milstein method under P
    '''
    fig, ax = plt.subplots(figsize=[11, 5])
    GBMpaths = np.zeros((N, int(1 / dt)))
    for j in range(N):
        GBMpaths[j] = GBM_M(mu, sigma, S0, dt)
        ax.plot(np.arange(int(1 / dt)), GBMpaths[j], '-o', label='$S_t$', ms=1, alpha=0.6)
    plt.show()

def GBM_EM_Q(r, sigma, S0, dt):
    '''
    Generate GBM under Q by Euler-Maruyama Approxixmation
    '''
    path = np.zeros(int(1 / dt))
    path[0] = S0
    for t in range(1, int(1 / dt)):
        deltaW = BIncr(dt)
        path[t] = np.exp(np.log(path[t-1]) + (r - 0.5 * sigma ** 2) * dt + sigma * deltaW)
    return path[int(1 / dt - 1)]

def GBM_Mil_Q(r, sigma, S0, dt):
    '''
    Generate GBM under Q by Milstein Approximiation
    '''
    path = np.zeros(int(1 / dt))
    path[0] = np.log(S0)
    for t in range(1, int(1 / dt)):
        deltaW = BIncr(dt)
        path[t] = path[t-1] + (r - 0.5 * sigma ** 2) * dt + sigma * deltaW + 0.5 * sigma * (deltaW ** 2 - dt)
    return np.exp(path[int(1 / dt) - 1])

# ------------------------ 2. Monte Carlo Functions (I) ------------------------
def MC_EuroCall_EM(r, sigma, S0, dt, K, n):
    '''
    Monte Carlo Simulation for european call by Euler-Maruyama.
    
    Version 1.0.
    '''
    sum = np.zeros(n)
    for j in range(n):
        sum[j] = sum[j-1] + np.max([GBM_EM_Q(r, sigma, S0, dt) - K, 0])
    return sum

def MC_EuroCall_M(r, sigma, S0, dt, K, n):
    '''
    Monte Carlo Simulation for european call by Milstein.
    
    Version 1.1
    '''
    sum = np.zeros(n)
    for j in range(n):
        sum[j] = sum[j-1] + np.max([GBM_Mil_Q(r, sigma, S0, dt) - K, 0])
    return sum

# ------------------------ 3. Monte Carlo Convergence ------------------------
def Conv(r, sigma, S0, dt, K, n, p):
    '''
    Evaluate convergence of MC algo
    '''
    x0 = MC_EuroCall_M(r, sigma, S0, dt, K, n)
    x1 = np.zeros(n)
    for j in range(n):
        x1[j] = np.absolute(np.exp(-r)*(x0[j])/(j + 1) - p)
        print("Iteration",j,":", x1[j])
    fig, ax = plt.subplots(figsize=[11, 5])
    ax.loglog(np.arange(n), x1, '-o', label='$S_t$', ms=1, alpha=0.6)
    plt.show()
    
def Conv0(r, sigma, S0, dt, K, n, k):
    '''
    Convergence of multiple MC paths
    '''
    fig, ax = plt.subplots(figsize=[11, 5])
    MCpaths = np.zeros((k, n))
    MCpaths1 = np.zeros((k, n))
    for i in range(k):
        MCpaths[i] = MC_EuroCall_EM(r, sigma, S0, dt, K, n)
    for i in range(k):
        for j in range(n):
            MCpaths1[i][j] = MCpaths[i][j]/(j + 1) - 19.386
        ax.plot(np.arange(n), MCpaths1[i], '-o', label='$S_t$', ms=1, alpha=0.6)
        ax.set_ylim([-10, 10]) # pyright: ignore[reportArgumentType]
    plt.show()

def var(r, sigma, S0, dt, K, n, p, k):
    '''
    Evaluate asymptotic properties of MC estimators; k independent MC sims
    '''
    s1 = np.zeros(k)
    s2 = np.zeros(k)
    for j in range(k):
        s1[j] = s1[j-1] + (MC_EuroCall_M(r, sigma, S0, dt, K, n)[int(n - 1)]/n - p) ** 2
    for i in range(k):
        s2[i] = ( s1[i] / (i + 1) ) ** 0.5
    fig, ax = plt.subplots(figsize=[11, 5])
    ax.plot(np.arange(k), s2, '-o', label='$S_t$', ms=1, alpha=0.6)
    plt.show()
    
def hist(r, sigma, S0, dt, K, n, p, k):
    '''
    Histogram of point estimate of k independent MC sims
    '''
    fig, ax = plt.subplots(figsize=[11, 5])
    h0 = np.zeros(k)
    for j in range(k):
        h0[j] = (MC_EuroCall_M(r, sigma, S0, dt, K, n)[int(n - 1)]/n - p) / p
    # plt.hist(h0, bins=100)
    # plt.show()
    print(n, dt, np.mean(h0),np.var(h0)**0.5)

# ----------------------- 3. Monte Carlo Functions (II) ------------------------
def MC_EuroCall_EM_F(r, sigma, S0, dt, K, n):
    '''
    Monte Carlo (EM).
    
    Version 2.0
    '''
    sum = np.zeros(n)
    for j in range(1,n):
        sum[j] = (sum[j-1] * (j - 1) + np.max([GBM_EM_Q(r, sigma, S0, dt) - K, 0])) / j
    return np.exp(-r) * sum[n-1]

# ----------------------------------- BREAK ------------------------------------
# ------------------- 4. Accelerated Diffusion Sims (III) ----------------------
def dW(T, dt):
    '''
    Generates Brownian Increment, variance dt
    '''
    return np.random.normal(0, dt ** 0.5, size=int(T / dt))

def Bpath_f(T, dt):
    '''
    Generates Brownian Path
    '''
    return np.cumsum(dW(T, dt))

def gbmP0(mu, sigma, S0, T, dt):
    '''
    Generates GBM sim under P
    '''
    dw = dW(T, dt)
    path = np.zeros(int(T / dt) + 1)
    path[0] = S0
    for t in range(1, int(T / dt) + 1):
        path[t] = path[t-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dw[t-1])
    return path

def gbmQ0(r, sigma, S0, T, dt):
    '''
    Generates GBM sim under Q
    '''
    dw = dW(T, dt)
    path = np.zeros(int(T / dt))
    path[0] = S0
    for t in range(1, int(T / dt)):
        path[t] = path[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dw[t-1])
    return path

def gbmPlot(mu, sigma, S0, T, dt):
    '''
    Plot GBM sample path
    '''
    path = gbmP0(mu, sigma, S0, T, dt)
    fig, ax = plt.subplots(figsize=[11, 5])
    ax.plot(np.arange(int(T / dt)), path, '-o', ms=0.1, alpha=0.6)
    plt.show()

# ---------------------- 5. Monte Carlo Functions (III) ------------------------
def MC(r, sigma, S0, T, dt, K, n):
    '''
    Monte Carlo European Option
    
    Version 3.0
    '''
    theo = ast.OptionEng(S0, K, r, T, 0, sigma, 'European', 'call', pr.BlackScholesEngine(), dt)
    p_mc = np.zeros(n)
    z = np.zeros(n)
    for j in range(n):
        p_mc[j] = gbmQ0(r, sigma, S0, T, dt)[int(1 / dt) - 1]
    a_MC = np.exp(-r * T) * np.sum(np.maximum((p_mc - K),z)) / n
    a_BS = theo.price()
    eps = abs(a_BS/a_MC-1)
    return ['BS', a_BS, 'MC', a_MC, 'eps', eps]

# ------------------- 6. Accelerated Diffusion Sims (IV) -----------------------
def dW1(T, dt, N):
    '''
    Draws N white noise increments with dt time step for T time.
    
    Returns matrix (array) with N rows and 1/dt columns.
    '''
    return np.random.normal(0, dt ** 0.5, size=(N, int(T / dt)))

def gbmP1(mu, sigma, S0, T, dt, N):
    """Draws N GBM's jointly under P.

    Args:
        mu (_type_): _description_
        
        sigma (_type_): _description_
        
        S0 (_type_): _description_
        
        T (_type_): _description_
        
        dt (_type_): _description_
        
        N (_type_): _description_

    Returns:
        np.array: Returns path, an (Nx1/dt) matrix with sample paths.
    """
    dw = dW1(T, dt, N)
    path = np.zeros((N, int(T / dt) + 1))
    path[:,0] = S0
    for t in range(1, int(T / dt) + 1):
        path[:,t] = path[:,t-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dw[:,t-1])
    return path

def gbmQ1(r, sigma, S0, T, dt, N):
    '''
    Draws N GBM's for Monte Carlo jointly under Q.
    
    Returns path, an (Nx1/dt) matrix with sample paths.
    '''
    dw = dW1(T, dt, N)
    path = np.zeros((N, int(T / dt) + 1))
    path[:,0] = S0
    for t in range(1, int(T / dt) + 1):
        path[:,t] = path[:,t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dw[:,t-1])
    return path

# ---------------------- 7. Monte Carlo Functions (IV) -------------------------
def MC0(r, sigma, S0, T, dt, K, N):
    '''
    Monte Carlo European Option
    
    Version 4.0
    '''
    theo = ast.OptionEng(S0, K, r, T, 0, sigma, 'European', 'call', pr.BlackScholesEngine(), dt)
    S_T = gbmQ1(r , sigma, S0, T, dt, N)[:,int(1 / dt)]
    a_MC = np.exp(-r * T) * np.sum(np.maximum((S_T - K), 0.0)) / N
    a_BS = theo.price()
    eps = abs(a_BS/a_MC-1)
    return ['BS', a_BS.round(6), 'MC', a_MC.round(6), 'eps', eps.round(6)]

# --------------------- 8. Accelerated Diffusion Sims (V) ----------------------
def dW2(T, dt, N):
    return (dt ** 0.5) * np.random.standard_normal(size=(N, int(T / dt)))

def gbmP2(mu, sigma, S0, T, dt, N):
    dw = dW2(T, dt, N)
    path = np.zeros((N, int(T / dt) + 1))
    path[:,0] = S0
    for t in range(1, int(T / dt) + 1):
        path[:,t] = path[:,t-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dw[:,t-1])
    return path

def gbmP3(mu, sigma, S0, T, dt, N):
    dw = dW2(T, dt, N)
    increments = (mu - 0.5 * sigma ** 2) * dt + sigma * dw
    log_paths = np.concatenate([np.zeros((N, 1)), np.cumsum(increments, axis = 1)], axis = 1)
    path = S0 * np.exp(log_paths)
    return path

# ------------------- 9. Multiproccesing Diffusion Sims (VI) -------------------
def gbm_chunk(args):
    mu, sigma, S0, T, dt, n_paths, seed = args
    rng = np.random.default_rng(seed)
    steps = int(T / dt)
    dw = rng.standard_normal(size=(steps, n_paths)) * np.sqrt(dt)
    increments = (mu - 0.5 * sigma**2) * dt + sigma * dw
    log_paths = np.vstack([np.zeros(n_paths), np.cumsum(increments, axis=0)])
    return S0 * np.exp(log_paths)

def gbmP4(mu, sigma, S0, T, dt, N):
    # Use all available CPU cores
    n_jobs = mp.cpu_count()
    
    base = N // n_jobs
    remainder = N % n_jobs
    chunk_sizes = [base] * n_jobs
    for i in range(remainder):
        chunk_sizes[i] += 1  # distribute remainder

    # Prepare arguments for each process
    seeds = np.random.SeedSequence().spawn(n_jobs)
    args_list = [(mu, sigma, S0, T, dt, chunk_sizes[i], seeds[i].entropy) for i in range(n_jobs)]
    
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(gbm_chunk, args_list)
    
    # Concatenate all chunks
    return np.hstack(results)

# --------------------------- 9. MJD Diffusion Sims ----------------------------
def ArrivalTime(lam, T=1.0):
    """
    Simulate jump arrival times in [0, T] for a Poisson process with rate lam (lambda).
    
    Samples interarrival times from exponential distribution and sums to compute arrival times.
    
    Parameters
    ----------
    lam : float
        Lambda - Intensity (rate) of the Poisson process.
    T : float, optional
        Time horizon (default=1.0).

    Returns
    -------
    np.ndarray
        Array of arrival times (sorted, strictly increasing).
    """
    t = 0.0
    arrivals = []
    
    while True:
        delta = np.random.default_rng().exponential(1.0 / lam)   # inter-arrival time
        t += delta
        if t < T:
            arrivals.append(t)
        else:
            break
    
    return np.array(arrivals)

def PoisProc(lam, dt):
    """
    Simulate a Poisson counting process over [0,1] with given arrival times.

    Parameters
    ----------
    lam : float
       Lambda - Intensity (rate) of the Poisson process.
    dt : float
        Time step size (fraction of year).

    Returns
    -------
    np.ndarray
        Array of counts at each time step, representing the cumulative number of jumps up to each time.
    """
    steps = int(1 / dt) + 1
    jt = ArrivalTime(lam)
    t_grid = np.linspace(0, 1, steps)
    counts = np.searchsorted(jt, t_grid, side='right')
    return counts

def JumpAmp(N, alpha, delta):
    """
    Generate amplitudes for N jumps using a lognormal distribution.

    Parameters
    ----------
    N : int
        Number of jumps.
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

def compPoisProc(lam, alpha, delta, dt):
    """
    Simulate a compound Poisson process over [0,1] with lognormal jump amplitudes.

    Parameters
    ----------
    lam : float
        Poisson process rate parameter (expected number of jumps).
    alpha : float
        Mean of the logarithm of jump size.
    delta : float
        Standard deviation of the logarithm of jump size.
    dt : float
        Time step size (fraction of year).

    Returns
    -------
    np.ndarray
        Array of process values at each time step (cumulative sum of jump amplitudes).
        Length is N+1, where N = int(1/dt): first element is 0, others are cumulative jump sums.
    """
    N = int(1 / dt) + 1
    # Draws Poisson process
    pois = PoisProc(lam, dt)
    # Draws jump amplitudes
    jump_amps = JumpAmp(pois[N - 1], alpha, delta)
    t_grid = np.linspace(0, 1, N)
    # Infers jump times (in grid not when drawn)
    jump_times = t_grid[np.where(pois[1:] > pois[:-1])[0] + 1]
    # Retrieve indices of rightmost jump included at each time
    jump_idx = np.searchsorted(jump_times, t_grid, side='right')
    # Cumulative sum of jump amplitudes for all jumps
    cumsum_amps = np.concatenate([[0], np.cumsum(jump_amps)])
    # maps summed jump amplitudes to path
    J = cumsum_amps[jump_idx]
    return J

def MJD(mu, sigma, S0, lam, alpha, delta, dt):
    """
    Simulate a Levy process corresponding to the Merton Jump Diffusion model by combining a geometric Brownian motion (GBM) with a compound Poisson jump process.

    Parameters
    ----------
    mu : float
        Drift of the GBM component.
    sigma : float
        Volatility of the GBM component.
    S0 : float
        Initial asset price.
    lam : float
        Poisson process rate parameter (expected number of jumps).
    alpha : float
        Mean of the logarithm of jump size.
    delta : float
        Standard deviation of the logarithm of jump size.
    dt : float
        Time step size (fraction of year).

    Returns
    -------
    np.ndarray
        Simulated asset price path under Merton Jump Diffusion model.
        Length is N+1, with S1[0] = S0 and S1[1:] subsequent prices.
    """
    Nsteps = int(1 / dt) + 1
    path_gbm = gbmP0(mu, sigma, S0, 1, dt)
    logS = np.log(path_gbm)
    J1 = compPoisProc(lam, alpha, delta, dt)
    logS1 = logS + J1
    S1 = np.exp(logS1)
    return S1

def LevyPlot(mu, sigma, S0, lam, alpha, delta, dt):
    '''
    Plot Levy Process
    '''
    S1 = MJD(mu, sigma, S0, lam, alpha, delta, dt)
    fig, ax = plt.subplots(figsize=[11, 5])
    ax.plot(np.arange(int(1 / dt) + 1), S1, '-o', label='$S_t$', ms=1, alpha=0.6)
    plt.show()
# ------------------------------ END ------------------------------