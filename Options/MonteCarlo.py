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

# ------------------------------ END ------------------------------