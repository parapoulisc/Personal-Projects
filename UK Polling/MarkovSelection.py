import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed
from tqdm import tqdm

import MarkovEstimate as me

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

# ------------------- Markov Model Selection -------------------
# Evaluation Metrics by fit
def MarkovEval(polls_df, T_stack, parties,
               start_date=None, end_date=None):
    """
    Evaluate given Markov model by returning metrics:
    - OneStep_MSE
    - OneStep_CE
    - RowDev_Max
    - ColDev_Max
    - RMSE / per party

    Returns
    -------
    metrics : dict
        Dictionary containing scalar and per-party metrics.
    """

    if start_date is None:
        start_date = polls_df.index.min()
    if end_date is None:
        end_date = polls_df.index.max()

    mse_list = []
    ce_list = []

    # --- Per-party squared errors ---
    per_party_sq_errors = {p: [] for p in parties}

    row_dev_list = []
    col_dev_list = []

    for t, (idx, T_t) in enumerate(T_stack.groupby(level=0)):
        if idx < pd.to_datetime(start_date) or idx > pd.to_datetime(end_date):
            continue

        # True vs predicted (convert shares to proportions)
        x_prev = polls_df.iloc[t-1][parties].to_numpy() / 100
        x_true = polls_df.iloc[t][parties].to_numpy() / 100
        K = len(parties)
        T_mat = T_t.to_numpy().reshape(K, K)
        x_pred = (T_mat @ x_prev).clip(1e-12, 1)

        # 1. MSE
        mse_list.append(np.mean((x_true - x_pred) ** 2))

        # 2. Cross-entropy (non-negative)
        ce_list.append(-np.sum(x_true * np.log(x_pred)))

        # 3. Per-party squared errors
        for j, p in enumerate(parties):
            per_party_sq_errors[p].append((x_true[j] - x_pred[j]) ** 2)

        # 4. Row/col deviations (use reshaped KxK matrix)
        row_dev_list.append(np.abs(T_mat.sum(axis=1) - 1))
        col_dev_list.append(np.abs(T_mat.sum(axis=0) - 1))

    # Flatten deviations
    row_dev_all = np.concatenate(row_dev_list) if row_dev_list else np.array([np.nan])
    col_dev_all = np.concatenate(col_dev_list) if col_dev_list else np.array([np.nan])

    # --- Aggregate metrics ---
    metrics = {
        "OneStep_MSE": np.mean(mse_list),
        "OneStep_CE": np.mean(ce_list),
        "RowDev_Max": np.max(row_dev_all),
        "ColDev_Max": np.max(col_dev_all),
    }

    # Add per-party RMSE to metrics dict
    for p in parties:
        metrics[f"RMSE_{p}"] = np.sqrt(np.mean(per_party_sq_errors[p]))

    return metrics

# Evaluation Metrics by Regression test
def MarkovEvalReg(polls_df, T_stack, parties,
                start_date=None, end_date=None, add_intercept=False):
    """
    Evaluate Markov model predictions with MSE/CE, per-party RMSE,
    row/col deviations, and regression of flows on daily poll changes.

    Returns
    -------
    metrics : dict
        Dictionary containing scalar metrics, per-party RMSE,
        and regression statistics (R², BIC, coefficients, etc.).
    """

    if start_date is None:
        start_date = polls_df.index.min()
    if end_date is None:
        end_date = polls_df.index.max()

    mse_list = []
    ce_list = []

    per_party_sq_errors = {p: [] for p in parties}
    row_dev_list = []
    col_dev_list = []

    # --- One-step predictions loop ---
    for t, (idx, T_t) in enumerate(T_stack.groupby(level=0)):
        if idx < pd.to_datetime(start_date) or idx > pd.to_datetime(end_date):
            continue

        # Transition matrix
        T_arr = T_t.to_numpy().reshape(len(parties), len(parties))

        x_prev = polls_df.iloc[t-1][parties].to_numpy() / 100
        x_true = polls_df.iloc[t][parties].to_numpy() / 100

        x_pred = (T_arr @ x_prev).clip(1e-12, 1)

        mse_list.append(np.mean((x_true - x_pred) ** 2))
        ce_list.append(-np.sum(x_true * np.log(x_pred)))

        for j, p in enumerate(parties):
            per_party_sq_errors[p].append((x_true[j] - x_pred[j]) ** 2)

        # Row/col sums (deviation from 1)
        row_dev_list.append(np.abs(T_arr.sum(axis=1) - 1))
        col_dev_list.append(np.abs(T_arr.sum(axis=0) - 1))

    # --- Aggregate metrics (MSE, CE, RMSE, row/col devs) ---
    metrics = {
        "OneStep_MSE": np.mean(mse_list),
        "OneStep_CE": np.mean(ce_list),
        "RowDev_Max": np.max(np.concatenate(row_dev_list)) if row_dev_list else np.nan,
        "ColDev_Max": np.max(np.concatenate(col_dev_list)) if col_dev_list else np.nan,
    }
    for p in parties:
        metrics[f"RMSE_{p}"] = np.sqrt(np.mean(per_party_sq_errors[p])) if per_party_sq_errors[p] else np.nan

    # --- Regression: ΔPoll ~ NetFlows ---
    from_polls = polls_df[parties].reindex(T_stack.index.get_level_values(0))
    poll_change = from_polls.diff()
    inflows, outflows, net_flows = me.FlowEst(T_stack, polls_df, parties=parties)

    poll_change = poll_change.reindex(net_flows.index)
    net = net_flows.reindex(poll_change.index)

    reg_stats = {}
    R2_vals, BIC_vals = [], []

    for p in parties:
        df = pd.concat([poll_change[p], net[p]], axis=1).dropna()
        if df.shape[0] < 3:
            reg_stats[p] = {"N": df.shape[0], "Coef": np.nan, "R2": np.nan, "BIC": np.nan}
            continue

        y = df.iloc[:, 0].to_numpy()
        X = df.iloc[:, 1].to_numpy()
        X_design = sm.add_constant(X) if add_intercept else X.reshape(-1, 1)

        model = sm.OLS(y, X_design).fit()

        coef = model.params[-1]
        R2 = model.rsquared
        BIC = model.bic

        reg_stats[p] = {"N": int(model.nobs), "Coef": float(coef), "R2": float(R2), "BIC": float(BIC)}

        R2_vals.append(R2)
        BIC_vals.append(BIC)

    # --- Regression summary ---
    metrics.update({
        "Mean_R2": np.mean(R2_vals) if R2_vals else np.nan,
        "Median_R2": np.median(R2_vals) if R2_vals else np.nan,
        "Mean_BIC": np.mean(BIC_vals) if BIC_vals else np.nan,
    })
    for p, stats in reg_stats.items():
        for k, v in stats.items():
            metrics[f"{p}_{k}"] = v

    return metrics

# Generate list of mesh points for grid search
def MeshPoints(lower_bound, upper_bound,
               spacing_factor=0.5, log_scale=False, precision=5):
    """
    Generate a full mesh grid of points between two bounds including bounds,
    returning floats rounded to the desired precision in significant digits.

    Parameters
    ----------
    lower_bound : float
        Minimum value of the range.
    upper_bound : float
        Maximum value of the range.
    spacing_factor : float
        Fractional spacing between points, e.g., 0.5 -> halve intervals, 0.25 -> quarter intervals.
    log_scale : bool
        If True, generate points in log-space instead of linear space.
    precision : int
        Number of significant digits for returned float values.

    Returns
    -------
    points : np.ndarray
        Sorted array of floats including bounds, rounded to `precision` significant digits.
    """
    if lower_bound >= upper_bound:
        raise ValueError("lower_bound must be less than upper_bound")
    if not (0 < spacing_factor < 1):
        raise ValueError("spacing_factor must be between 0 and 1")

    n_intervals = max(1, int(np.round(1 / spacing_factor)))

    if log_scale:
        log_lower = np.log10(lower_bound)
        log_upper = np.log10(upper_bound)
        points = np.logspace(log_lower, log_upper, n_intervals + 1)
    else:
        points = np.linspace(lower_bound, upper_bound, n_intervals + 1)

    # Round to chosen number of significant digits
    def round_sig(x, sig):
        if x == 0:
            return 0.0
        else:
            return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

    points = np.array([round_sig(p, precision) for p in points])
    return points

# Utility for grid search
def GridEval(polls_df, parties,
             window, alpha, lam, step=1,
             start_date=None, end_date=None, use_reg=False):
    """
    Run rolling Markov estimation for given parameters and evaluate.

    Parameters
    ----------
    polls_df : pd.DataFrame
        Polling data.
    parties : list of str
        Party names.
    window : int
        Rolling window size.
    alpha : float
        Elastic Net mixing parameter.
    lam : float
        Regularization parameter.
    step : int
        Rolling step size.
    start_date, end_date : optional
        Restrict evaluation to date range.
    use_reg : bool
        If True, evaluate using regression-based metrics.

    Returns
    -------
    pd.DataFrame
        One row indexed by (window, alpha, lam) with evaluation metrics.
    """
    T_stack = me.MarkovEstRoll(
        polls_df=polls_df,
        parties=parties,
        window=window,
        step=step,
        alpha=alpha,
        lam=lam,
        start_date=start_date,
        end_date=end_date
        )

    if use_reg:
        metrics = MarkovEvalReg(polls_df, T_stack, parties, start_date=start_date, end_date=end_date)
    else:
        metrics = MarkovEval(polls_df, T_stack, parties, start_date=start_date, end_date=end_date)

    row = pd.DataFrame([metrics], index=pd.MultiIndex.from_tuples(
        [(window, alpha, lam)],
        names=["Window", "Alpha", "Lambda"]
    ))
    return row

# Grid Search for Markov Model
def GridSearch(polls_df, parties,
               W_list, a_list, l_list, step,
               start_date=None, end_date=None):
    """
    Perform a grid search over Markov model hyperparameters.

    Parameters
    ----------
    polls_df : pd.DataFrame
        DataFrame of polling data.
    parties : list of str
        List of party names.
    W_list : list of int
        List of window sizes to try.
    a_list : list of float
        List of alpha (Elastic Net mixing) values to try.
    l_list : list of float
        List of lambda (regularization) values to try.
    step : int
        Step size for rolling window.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of evaluation metrics for each parameter combination.
    """
    results = []
    for W in W_list:
        for a in a_list:
            for l in l_list:
                row = GridEval(
                    polls_df, parties, W, a, l, step=step,
                    start_date=start_date, end_date=end_date
                )
                results.append(row)
    return pd.concat(results)

# Perform grid search in parallel
def GridSearchParallel(polls_df, parties,
                       W_list, a_list, l_list, step=1,
                       start_date=None, end_date=None,
                       n_jobs=-1, use_reg=False):
    """
    Parallel grid search over (W, alpha, lambda).

    Arguments:
      polls_df (pd.Dataframe)  : DataFrame of smoothed polls
      parties (list)   : list of parties
      W_list (list)    : list of window lengths
      a_list (list)    : list of alphas
      l_list (list)    : list of lambdas
      step (int)     : rolling step size
      start_date (str): optional start date
      end_date (str)  : optional end date
      n_jobs (int)    : number of parallel workers (-1 = all cores). Defaults to this.
      use_reg (bool) : whether to use regression-based evaluation (evaluate_markov_model_reg). Defaults to False.

    Returns:
      DataFrame indexed by (Window, Alpha, Lambda) with evaluation metrics.
    """

    # Build parameter grid
    param_grid = [(W, a, l) for W in W_list for a in a_list for l in l_list]
    
    GridSearchResults = pd.read_csv('/Users/constantineparapoulis/Documents/Projects/UK Polling Stuff/Grid Search Output/GridSearch_reg.csv')
    
    existing = set(zip(GridSearchResults["Window"], 
                    GridSearchResults["Alpha"], 
                    GridSearchResults["Lambda"]))
    param_grid = [params for params in param_grid if params not in existing]    

    # Run in parallel with progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(GridEval)(
            polls_df, parties, W, a, l, step=step,
            start_date=start_date, end_date=end_date, use_reg=use_reg
        )
        for (W, a, l) in tqdm(param_grid, desc="Grid Search", unit="model")
    )

    # Combine all results
    return pd.concat(results)

# --------------- Markov Model Selection: Graphics ---------------
# Visualisation
def GridSearchPlot(results_df, kind="contour"):
    """
    Visualize grid search results.
    
    For each window, creates two rows of subplots:
      - First row: OneStep_MSE (colormap reversed, brighter colors for smaller values)
      - Second row: OneStep_CE (normal colormap)
    Parameters
    ----------
    results_df : DataFrame
        Must contain columns: ["Window", "Alpha", "Lambda", "OneStep_MSE", "OneStep_CE"]
    kind : str
        Type of plot: {"surface", "contour"}.
    """
    # Prepare data
    df = results_df.reset_index(drop=True).copy()
    df["Lambda_log10"] = np.log10(df["Lambda"])
    # Prepare both metrics
    df["OneStep_MSE_plot"] = df["OneStep_MSE"]
    df["OneStep_CE_plot"] = df["OneStep_CE"]

    metrics = [("OneStep_MSE_plot", "OneStep_MSE", plt.get_cmap("viridis_r")), ("OneStep_CE_plot", "OneStep_CE", plt.get_cmap("viridis"))]
    metric_labels = ["OneStep_MSE", "OneStep_CE"]
    windows = df["Window"].unique()
    nW = len(windows)
    nrows = 2
    fig, axes = plt.subplots(
        nrows, nW, figsize=(6 * nW, 5 * nrows),
        subplot_kw={"projection": "3d"} if kind != "contour" else {}
    )
    # axes shape: (2, nW)
    if nW == 1:
        axes = np.array(axes).reshape(nrows, 1)

    for j, (col, label, cmap) in enumerate(metrics):
        for i, W in enumerate(windows):
            ax = axes[j, i]
            sub = df[df["Window"] == W]
            X = sub["Alpha"]
            Y = sub["Lambda_log10"]
            Z = sub[col]
            if kind == "surface":
                # Pivot so that index=Alpha (Y-axis), columns=Lambda_log10 (X-axis)
                pivot = sub.pivot(index="Alpha", columns="Lambda_log10", values=Z.name)
                Xg, Yg = np.meshgrid(pivot.columns, pivot.index)
                Zg = pivot.values

                surf = ax.plot_surface(Xg, Yg, Zg, cmap=cmap, edgecolor="none", alpha=0.9)
                fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
                ax.set_xlabel("log10(Lambda)")
                ax.set_ylabel("Alpha")
                ax.set_zlabel(label)
                ax.set_title(f"{label} (W={W})")
            elif kind == "contour":
                # Pivot so that index=Alpha (Y-axis), columns=Lambda_log10 (X-axis)
                pivot = sub.pivot(index="Alpha", columns="Lambda_log10", values=Z.name)
                Xg, Yg = np.meshgrid(pivot.columns, pivot.index)
                Zg = pivot.values

                c = ax.contourf(Xg, Yg, Zg, levels=20, cmap=cmap)
                fig.colorbar(c, ax=ax)
                ax.set_xlabel("log10(Lambda)")
                ax.set_ylabel("Alpha")
                ax.set_title(f"{label} (W={W})")

    plt.tight_layout()
    plt.show()

# Visualisation - expanded reg test
def GridSearchRegPlot(results_df, metrics,
                      window_limit=None, alpha_limit=None, lambda_limit=None,
                      apply_threshold=False, threshold=20,
                      threshold_relative=True):
    """
    Visualize grid search regression results for multiple metrics across (Alpha, Lambda, Window).

    Parameters
    ----------
    results_df : DataFrame
        Must contain columns: ["Window", "Alpha", "Lambda", ...metrics...].
    metrics : list of str
        List of metric column names to plot.
    apply_threshold : bool
        If True, highlight threshold contour.
    threshold : float
        If threshold_relative=True, percentile cutoff (e.g., 20 -> top/bottom 20%).
        If threshold_relative=False, absolute cutoff value.
    threshold_relative : bool
        If True, interpret `threshold` as percentile; if False, as absolute cutoff value.
    """
    special_metrics = [
        'OneStep_CE', 'RowDev_Max', 'ColDev_Max', 'Mean_R2', 'Median_R2', 'Mean_BIC',
        'Con_Coef', 'Con_R2', 'Con_BIC',
        'Lab_Coef', 'Lab_R2', 'Lab_BIC',
        'LD_Coef', 'LD_R2', 'LD_BIC',
        'Green_Coef', 'Green_R2', 'Green_BIC',
        'FarRight_Coef', 'FarRight_R2', 'FarRight_BIC',
        'Others_exp_Coef', 'Others_exp_R2', 'Others_exp_BIC'
    ]

    for m in metrics:
        if m not in results_df.columns:
            raise ValueError(f"Metric '{m}' not in DataFrame. Available: {list(results_df.columns)}")

    df = results_df.reset_index(drop=True).copy()
    df["Lambda_log10"] = np.log10(df["Lambda"])

    windows = df["Window"].unique()
    if window_limit is not None:
        windows = [j for j in windows if window_limit[0] <= j <= window_limit[1]]
    windows.sort()
    
    nW, nM = len(windows), len(metrics)
    fig, axes = plt.subplots(nM, nW, figsize=(6 * nW, 4 * nM))

    if nM == 1: axes = np.expand_dims(axes, 0)
    if nW == 1: axes = np.expand_dims(axes, 1)

    for i, metric in enumerate(metrics):
        sub_df = df.copy()
        for j, W in enumerate(windows):
            ax = axes[i, j]
            sub = sub_df[sub_df["Window"] == W].copy()

            # --- Compute cutoff for highlighting ---
            cutoff = None
            cutoff_label = None
            if apply_threshold and not sub.empty:
                if threshold_relative:
                    if metric in special_metrics:
                        cutoff = np.nanpercentile(sub[metric], 100 - threshold)
                        cutoff_label = f"Top {threshold}%"
                    else:
                        cutoff = np.nanpercentile(sub[metric], threshold)
                        cutoff_label = f"Bottom {threshold}%"
                else:
                    cutoff = threshold
                    cutoff_label = f"Cutoff = {threshold}"

            # Extract x, y, z
            x, y, z = sub["Alpha"], sub["Lambda_log10"], sub[metric]

            # Check regular grid
            unique_alpha = np.sort(sub["Alpha"].unique())
            unique_lambda = np.sort(sub["Lambda_log10"].unique())
            mesh_size = len(unique_alpha) * len(unique_lambda)
            mesh = pd.MultiIndex.from_product([unique_alpha, unique_lambda], names=["Alpha", "Lambda_log10"]) # type: ignore
            mesh_df = pd.DataFrame(index=mesh).reset_index()
            merged = pd.merge(mesh_df, sub[["Alpha", "Lambda_log10", metric]], 
                              on=["Alpha", "Lambda_log10"], how="left")
            grid_is_regular = (len(sub) == mesh_size) and (not merged[metric].isnull().any())

            # Choose colormap
            cmap = "viridis" if metric in special_metrics else "viridis_r"

            if grid_is_regular:
                pivot = sub.pivot(index="Lambda_log10", columns="Alpha", values=metric)
                X, Y = np.meshgrid(pivot.columns, pivot.index)
                Z = pivot.values
                c = ax.contourf(X, Y, Z, levels=20, cmap=cmap)

                if cutoff is not None:
                    ax.contour(X, Y, Z, levels=[cutoff], colors="black", linewidths=1.5)
                    ax.text(X.max(), Y.max(), cutoff_label,
                            ha="right", va="bottom", fontsize=7, color="black")
            else:
                c = ax.tricontourf(x, y, z, levels=20, cmap=cmap)

                if cutoff is not None:
                    ax.tricontour(x, y, z, levels=[cutoff], colors="black", linewidths=1.5)
                    ax.text(max(x), max(y), cutoff_label,
                            ha="right", va="bottom", fontsize=7, color="black")

            fig.colorbar(c, ax=ax)
            if alpha_limit is not None:
                ax.set_xlim(alpha_limit[0], alpha_limit[1])
            if lambda_limit is not None:
                ax.set_ylim(lambda_limit[0], lambda_limit[1])
            ax.set_xlabel("Alpha")
            ax.set_ylabel("log10(Lambda)")
            ax.set_title(f"{metric} (W={W})")

    plt.tight_layout()
    plt.show()
    
def GridSearchTradeoffPlot(results_df, x_cols, y_cols):
    """
    Create 4 scatter plots in a 1x4 subplot layout.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    x_cols : list of str
        List of 4 column names to use as x-axis variables.
    y_cols : list of str
        List of 4 column names to use as y-axis variables.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)

    for i in range(4):
        ax = axes[i]
        ax.scatter(results_df[x_cols[i]], results_df[y_cols[i]], alpha=0.7)
        ax.set_xlabel(x_cols[i])
        ax.set_ylabel(y_cols[i])

    plt.tight_layout()
    plt.show()
    
# ------------------- END -------------------