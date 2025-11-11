import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import cvxpy as cp
plt.rcParams.update({
    'figure.dpi': 600,
    'figure.figsize': (10, 6),
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 0.5,
    'lines.linestyle': '-',
    'axes.grid': True,
    'grid.linestyle': '--',
    'legend.fontsize': 12
})

# -------------------------- Markov Model Estimation ---------------------------
# Transition matrix diagnostics
def TrMatDiag(T_stack,
              parties = ["Con","Lab","LD","Green","FarRight","Others_exp"]):
    """
    Diagnostics for row-stochastic transition matrices.
    Columns = MultiIndex (to, from)
    Rows = dates
    Outflows (rows for each from-party) should sum to 1
    Inflows (columns for each to-party) unconstrained
    """
    deviation_stats = []

    for j in parties:
        # --- Outflows = sum across row for fixed from_party j ---
        outflow_cols = [col for col in T_stack.columns if col[1] == j]
        outflow_sum = T_stack[outflow_cols].sum(axis=1)
        outflow_dev = (outflow_sum - 1).abs()

        # --- Inflows = sum across row for fixed to_party j ---
        inflow_cols = [col for col in T_stack.columns if col[0] == j]
        inflow_sum = T_stack[inflow_cols].sum(axis=1)
        inflow_dev = (inflow_sum - 1).abs()

        deviation_stats.append({
            "Party": j,
            # Row deviations (outflows)
            "Outflow_Mean": outflow_dev.mean(),
            "Outflow_Min": outflow_dev.min(),
            "Outflow_Q25": outflow_dev.quantile(0.25),
            "Outflow_Q50": outflow_dev.median(),
            "Outflow_Q75": outflow_dev.quantile(0.75),
            "Outflow_Max": outflow_dev.max(),
            # Column deviations (inflows)
            "Inflow_Mean": inflow_dev.mean(),
            "Inflow_Min": inflow_dev.min(),
            "Inflow_Q25": inflow_dev.quantile(0.25),
            "Inflow_Q50": inflow_dev.median(),
            "Inflow_Q75": inflow_dev.quantile(0.75),
            "Inflow_Max": inflow_dev.max()
        })

    deviation_df = pd.DataFrame(deviation_stats).set_index("Party")

    print('----------------- Transition Matrix Diagnostics -----------------')
    print('--------------------------- Outflows (rows) --------------------')
    print(deviation_df.iloc[:, 0:6].round(4))
    print('--------------------------- Inflows (columns) ------------------')
    print(deviation_df.iloc[:, 6:12].round(4))
    print('-----------------------------------------------------------------')

# Transition matrix estimation - single period
def MarkovEst(polls_df,
              parties = ["Con","Lab","LD","Green","FarRight","Others_exp"],
              weights = None, alpha = 0.5, lam = 1e-4,
              start_date=None, end_date=None):
    """
    Estimate a column-stochastic transition matrix T (rows=to, cols=from).
    
    Estimation performed by Elastic Net regularisation.
    """
    df = polls_df
    if start_date is None:
        start_date = df.index.min()
    if end_date is None:
        end_date = df.index.max()
    df = df.loc[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

    P = df[parties].to_numpy(dtype=float)
    n_obs, K = P.shape

    # Normalise input poll rows
    row_sums = P.sum(axis=1, keepdims=True)
    good = (row_sums.squeeze() > 0)
    P[good] = P[good] / row_sums[good]

    # Lagged design matrices
    X = P[:-1, :]
    Y = P[1:, :]
    n = X.shape[0]

    # weights
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape[0] != n:
        raise ValueError("weights length must equal number of transitions (T-1)")

    # Row weight
    W_sqrt = np.sqrt(w)[:, None]
    Xw = W_sqrt * X
    Yw = W_sqrt * Y

    # Decision variable: rows=to, cols=from
    T = cp.Variable((K, K), nonneg=True)

    # Objective function
    fit = cp.sum_squares(Yw - Xw @ cp.transpose(T))
    enet = alpha * cp.norm1(T) + (1 - alpha) * cp.sum_squares(T)

    constraints = [cp.sum(T, axis=0) == np.ones(K)]
    prob = cp.Problem(cp.Minimize(0.5 * fit + lam * enet), constraints) # type: ignore
    prob.solve(solver=cp.OSQP, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
        except cp.SolverError:
            prob.solve(solver=cp.SCS, verbose=False)

    if T.value is None:
        raise RuntimeError(f"Elastic Net QP failed, status {prob.status}")

    T_val = np.clip(T.value, 0.0, 1.0)
    return pd.DataFrame(T_val, index=parties, columns=parties)

# Transition matrix estimation - rolling
def MarkovEstRoll(polls_df,
                  parties = ["Con","Lab","LD","Green","FarRight","Others_exp"],
                  window = 60, step = 1, alpha= 0.5, lam = 1e-4,
                  diag = False, start_date=None, end_date=None):
    """
    Rolling-window estimation of row-stochastic time-varying Markov transition matrices using an Elastic Net regularisation via constrained Quadratic Program (QP).

    This function estimates transition matrices for each rolling window of polls, enforcing:
    - Non-negativity (probabilities >= 0)
    - Row-stochasticity (rows sum to 1; outflows from each party sum to 1)
    - Elastic Net regularization to balance sparsity and shrinkage (controlled by alpha and lam)

    Convention:
    -----------
    - Row-stochastic: each row corresponds to a 'from_party' and sums to 1.
    - Columns are a MultiIndex: (to_party, from_party)
    - Rows are dates corresponding to the last date in each rolling window.

    Inputs:
    -------
    polls_df : pd.DataFrame
        T x K DataFrame of party shares (rows sum ~1), indexed by date.
    parties : list of str
        Names of parties (columns in polls_df) to include in estimation.
    window : int
        Length of the rolling window (number of time steps / days) for each transition matrix.
    step : int
        Step size for moving the rolling window forward. Default = 1.
    alpha : float [0,1]
        Mixing parameter for Elastic Net:
        - alpha = 1 : Lasso (L1)
        - alpha = 0 : Ridge (L2)
        - intermediate values combine L1 and L2 penalties.
    lam : float
        Regularization strength for Elastic Net penalty.
    diag : boolean
        Whether to produce transition matrix diagnostics on row/column sums; best/worst case.
    start_date : optional
        Restrict estimation to polls_df within this start date.
    end_date : optional
        Restrict estimation to polls_df within this end date.

    Outputs:
    --------
    T_stack : pd.DataFrame
        A time-indexed DataFrame of rolling transition matrices:
        - Index: dates (end of each rolling window)
        - Columns: MultiIndex (to_party, from_party)
        - Each row is a row-stochastic transition matrix for the corresponding window.

    Behavior:
    ---------
    1. For each rolling window of length `window`:
        - Estimate a K x K transition matrix T using `estimate_markov_qp_enet()`.
        - T satisfies non-negativity and row-stochastic constraints.
        - Elastic Net penalty helps regularize noisy or collinear transitions.
    2. Store the flattened matrix for that window.
    3. After rolling through the entire series with the given `step`, reconstruct
       T_stack as a time-indexed DataFrame with MultiIndex columns (to, from).
    4. Calls `T_matrix_diagnostics()` to display deviations from perfect row- and column-stochasticity.
    
    Notes:
    ------
    - Row-stochastic convention: the sum of probabilities across each row (for each 'from_party') is 1.
    - Column sums (inflows to a party) are generally unconstrained.
    """

    df = polls_df
    if start_date is None:
        start_date = df.index.min()
    if end_date is None:
        end_date = df.index.max()
    df = df.loc[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

    T_list = []
    idx_list = []

    for start in range(0, len(df)-window+1, step):
        end = start + window
        sub = df.iloc[start:end]

        T_est = MarkovEst(polls_df=sub, parties=parties,
                          alpha=alpha, lam=lam
                          )

        # Flatten KxK matrix row-wise (to-party contiguous)
        T_list.append(T_est.to_numpy().flatten(order='F'))
        # column-stochastic: 'F' keeps columns together
        idx_list.append(df.index[end-1])

    columns = pd.MultiIndex.from_product([parties, parties], names=["to","from"])

    T_stack = pd.DataFrame(T_list, index=idx_list, columns=columns)

    # Optional row/col sum diagnostics
    if diag == True:
        TrMatDiag(T_stack, parties)

    return T_stack

# ------------------------------ Voter Flow Model ------------------------------
# Model Estimation
def FlowEst(T_stack, smoothed_polls,
            parties = ["Con","Lab","LD","Green","FarRight","Others_exp"]):
    """
    Compute inflows, outflows, and net flows for each party.

    Parameters
    ----------
    T_stack : pd.DataFrame
        Time-indexed DataFrame of transition matrices, MultiIndex (to, from).
    smoothed_polls : pd.DataFrame
        Time series of party vote shares (indexed by date).
    parties : list of str
        Party names (consistent across T_stack and smoothed_polls).

    Returns
    -------
    inflows : pd.DataFrame
        Share of electorate flowing into each party at each time.
    outflows : pd.DataFrame
        Share of electorate flowing out of each party at each time.
    net_flows : pd.DataFrame
        inflows - outflows for each party at each time.
    """
    inflows = pd.DataFrame(index=T_stack.index, columns=parties, dtype=float)
    outflows = pd.DataFrame(index=T_stack.index, columns=parties, dtype=float)

    for t in T_stack.index:
        T = T_stack.loc[t].unstack().loc[parties, parties]  # KxK (from=row, to=col)
        prev_share = smoothed_polls.loc[t, parties].values  # electorate at t

        # Flow matrix: row=from, col=to
        flow_matrix = np.diag(prev_share) @ T.values
        flow_df = pd.DataFrame(flow_matrix, index=parties, columns=parties)

        inflows.loc[t] = flow_df.sum(axis=0)   # sum over from-parties → inflows to each party
        outflows.loc[t] = flow_df.sum(axis=1)  # sum over to-parties → outflows from each party

    net_flows = inflows - outflows
    return inflows, outflows, net_flows

# Plot flows for given party
def FlowPlot(inflows, outflows, net_flows, party,
             start_date=None, end_date=None):
    """
    Plot inflows, outflows, and net flows for a given party.

    Parameters
    ----------
    inflows, outflows, net_flows : pd.DataFrame
        DataFrames returned by compute_flows().
    party : str
        Party name to plot.
    start_date, end_date : str or pd.Timestamp, optional
        Date limits for x-axis.
    """
    if start_date is None:
        start_date = str(inflows.index.min())
    if end_date is None:
        end_date = str(inflows.index.max())
    plt.figure(figsize=(10,5))
    plt.plot(inflows.index, inflows[party], label="Inflows")
    plt.plot(outflows.index, outflows[party], label="Outflows")
    plt.plot(net_flows.index, net_flows[party], label="Net flow", linestyle="--")
    plt.title(f"Voter flows for {party}")
    plt.ylabel("Share of electorate")
    plt.legend()
    if start_date and end_date:
        plt.xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))
    plt.show()

# Plot net flows compared to differenced polling for given party
def FlowPlotPolls(smoothed_polls, inflows, outflows, net_flows, parties = None,
                  start=None, end=None, diff=1):
    """
    Compare polling change with estimated voter flows for all parties.

    Parameters
    ----------
    smoothed_polls : pd.DataFrame
        Polling time series (shares, rows indexed by date).
    inflows, outflows, net_flows : pd.DataFrame
        Flow series from compute_scaled_flows().
    start, end : str or pd.Timestamp, optional
        Date range for x-axis.
    parties : list of str, optional
        List of parties to plot. Defaults to intersection of smoothed_polls and net_flows columns.
    """
    # Parties to plot
    if parties is None:
        parties = [p for p in smoothed_polls.columns if p in net_flows.columns]
    n_parties = len(parties)
    if n_parties == 0:
        return

    # Date range
    if start or end:
        poll = smoothed_polls.loc[start:end, parties]
        net = net_flows.loc[start:end, parties]
    else:
        poll = smoothed_polls[parties]
        net = net_flows[parties]

    # Difference polling data
    poll_change = poll.diff(diff)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), sharex=True)
    axes = axes.flatten()
    for i, party in enumerate(parties):
        ax = axes[i]
        ax.plot(poll_change.index, poll_change[party], label="Δ Poll share", color="black", linewidth=2)
        ax.plot(net.index, net[party], label="Net flow", linestyle="--", color="red")
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_ylabel("Share of electorate")
        ax.set_title(f"Change in polling vs estimated net flows ({party})")
        ax.legend()

    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()
    
    # Regression Analysis
    print("{:<10} {:>12} {:>12} {:>12}".format("Party", "Coef", "t-stat", "R^2"))
    print("-" * 50)
    for party in parties:
        poll_series = poll_change[party]
        net_series = net[party]
        aligned = pd.concat([poll_series, net_series], axis=1, keys=["poll_change", "net_flow"]).dropna()
        if len(aligned) < 2:
            coef = np.nan
            tstat = np.nan
            rsq = np.nan
        else:
            X = aligned["net_flow"].values
            y = aligned["poll_change"].values
            model = sm.OLS(y, X).fit()
            coef = model.params[0]
            tstat = model.tvalues[0]
            rsq = model.rsquared
        print("{:<10} {:>12.4f} {:>12.4f} {:>12.4f}".format(party, coef, tstat, rsq))

# ------------------------------------ END -------------------------------------