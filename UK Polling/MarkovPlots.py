import matplotlib.pyplot as plt
import pandas as pd
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

import PollPlots as pp

# --------------------------- Markov Model Graphics ----------------------------
# Plot a single transition over time
def PlotTrMat(T_stack: pd.DataFrame,
              from_party, to_party, start_date=None, end_date=None):
    """
    Plot time series of the single matrix entry T_{to,from}.
    Modelled by row-stochastic convention.
    """
    if start_date is None:
        start_date = str(T_stack.index.min())
    if end_date is None:
        end_date = str(T_stack.index.max())
    
    # Event markers
    event_dates = pp.GetEvents(start_date, end_date)

    s = T_stack[(from_party, to_party)]

    fig, ax = plt.subplots(figsize=(10,3))
    s.plot(ax=ax, marker="o", ms=0.75, linestyle="-")
    ax.axhline(0, color="k", linewidth=0.4, linestyle="--")
    ax.set_ylabel("Probability")
    ax.set_title(f"Flow from {from_party} → {to_party}")
    ax.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date)) # type: ignore

    for name, d in event_dates.items():
        ax.axvline(x=d, color="red", linestyle="--", linewidth=0.5) # type: ignore
        ax.text(d, ax.get_ylim()[1]*0.95, name, rotation=90, ha="right", va="top", # type: ignore
                fontsize=5, color="red")
    
    plt.tight_layout()
    plt.show()

# Return axis for subplots
def ReturnTrMat(T_stack: pd.DataFrame,
                from_party, to_party,
                start_date=None, end_date=None, ax=None):
    """
    Returns axis after plotting T_{to,from}.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,3))
    if start_date is None:
        start_date = str(T_stack.index.min())
    if end_date is None:
        end_date = str(T_stack.index.max())
    s = T_stack[(from_party, to_party)]
    ax.plot(s.index, s.to_numpy(), marker="o", ms=1)
    ax.axhline(0, color="k", linewidth=0.4, linestyle="--")
    ax.set_title(f"{from_party} → {to_party}")
    ax.set_ylabel("Probability")
    ax.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date)) # type: ignore

    # Event markers
    event_dates = pp.GetEvents(start_date, end_date)

    for name, d in event_dates.items():
        ax.axvline(x=d, color="red", linestyle="--", linewidth=0.5) # type: ignore
        ax.text(d, ax.get_ylim()[1]*0.95, name, rotation=90, ha="right", va="top", fontsize=5, color="red") # type: ignore

    return ax

# Plot flows out from a given party
def PlotTrMatOut(T_stack: pd.DataFrame,
                 from_party,
                 start_date=None, end_date=None,
                 parties=["Con","Lab","LD","Green","FarRight","Others_exp"]):
    """
    Plot all flows out from a given party (row-stochastic convention).
    """
    if start_date is None:
        start_date = str(T_stack.index.min())
    if end_date is None:
        end_date = str(T_stack.index.max())
    for to_party in parties:
        PlotTrMat(T_stack, from_party=from_party, to_party=to_party,
                  start_date=start_date, end_date=end_date)

# Plot flows into a given party
def PlotTrMatIn(T_stack: pd.DataFrame,
                to_party,
                start_date=None,
                end_date=None,
                parties=["Con","Lab","LD","Green","FarRight","Others_exp"]):
    """
    Plot all flows out from a given party (row-stochastic convention).
    """
    if start_date is None:
        start_date = str(T_stack.index.min())
    if end_date is None:
        end_date = str(T_stack.index.max())
    for from_party in parties:
        PlotTrMat(T_stack, from_party=from_party, to_party=to_party,
                          start_date=start_date, end_date=end_date)

# Plot flow pairs for all parties
def PlotTrMatInOut(T_stack: pd.DataFrame,
                   start_date=None, end_date=None,
                   parties=["Con","Lab","LD","Green","FarRight","Others_exp"]):
    """
    Plots all elements of transition matrix time series.
    
    Flows of each possible party combination. 

    Args:
        T_stack (pd.DataFrame): Time series of transition matrix
        start_date (_type_, optional): _description_. Defaults to str(daily.index.min()).
        end_date (_type_, optional): _description_. Defaults to str(daily.index.max()).
        parties (list, optional): _description_. Defaults to ["Con","Lab","LD","Green","FarRight","Others_exp"].
    """
    if start_date is None:
        start_date = str(T_stack.index.min())
    if end_date is None:
        end_date = str(T_stack.index.max())
        
    pairs = [(f, t) for f in parties for t in parties]
    n = len(pairs)
    n_cols, n_rows = len(parties), len(parties)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 3*n_rows), sharex=True)
    axes = axes.flatten()

    for i, (from_party, to_party) in enumerate(pairs):
        ReturnTrMat(T_stack, from_party, to_party, start_date, end_date, ax=axes[i])

    # Remove unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# ------------------------------------ END -------------------------------------