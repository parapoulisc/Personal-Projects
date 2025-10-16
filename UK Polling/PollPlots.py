import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional
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

# ------------------------------- Polling Plots --------------------------------
# Define & filter event dates
def GetEvents(start_date=None, end_date=None):
    """Defines list of events in timeseries which will be marked in graphics.
    
    Filters based on start/end dates to match graphics bounds.
    
    Args:
        start_date (str, optional): Defaults to None.
        end_date (str, optional): Defaults to None.

    Returns:
        event_dates (dict): Filtered event dates dictionary.
    """
    event_dates = {"Starmer (E)": "2024-07-04",
                "Autumn Budget 2024": "2024-10-30",
                "Spring Statement 2025": "2025-03-26",
                "Local Elections": "2025-05-01",
                "Southport Riots": "2024-07-30",
                "Summer Recess End": "2025-09-01",
                "Summer Recess Start": "2025-07-22",
                "GE Announced": "2024-05-22",
                "Sunak": "2022-10-25",
                "Truss": "2022-09-06",
                "Johnson II (E)": "2019-12-12",
                "Johnson I": "2019-07-23",
                "May II (E)": "2017-06-08",
                "May I": "2016-07-13",
                "Cameron II (E)": "2015-05-07",
                "Cameron I/Clegg (E)": "2010-05-10",
                "Brown": "2007-06-27",
                "Blair III (E)": "2005-05-05",
                "Blair II (E)": "2001-06-07",
                "Blair I (E)": "1997-05-01"
                }
    event_dates = {name: pd.to_datetime(d) for name, d in event_dates.items()}
    event_dates = {name: d for name, d in event_dates.items() if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)} # type: ignore
    
    return event_dates

# Polling plot functions
def PlotPolls(smoothed_polls: pd.DataFrame,
              start_date=None, end_date=None,
              support_threshold=1.0, tight = False, tight_margin = 0.05,
              daily = None):
    
    """Plots time series of smoothed party support and polling aggregates.

    Args:
        smoothed_polls (pd.DataFrame): Daily polling figures, following two-pass smoothing by Kalman Filter. Contains values for parties and aggregates.
        start_date (str, optional): Starting date for graphics. Defaults to earliest time series value.
        end_date (str, optional): Ending date for graphics. Defaults to latests time series value.
        support_threshold (float, optional): Excludes time series of parties with insignificant support. Defaults to 1.0%.
        tight (boolean, optional): Plots with tighter margins, use for granularity. Defaults to False. Defaults to 0.05.
        daily (pd.Dataframe, optional): Raw polls for infering individial parties to be plotted. Set to daily = daily when calling function for correct graphics. Defaults to None.
    """
    
    if start_date is None:
        start_date = str(smoothed_polls.index.min())
    if end_date is None:
        end_date = str(smoothed_polls.index.max())

    # Retrieve dictionary of events
    event_dates = GetEvents(start_date,end_date)

    # Support threshold
    mask = (smoothed_polls.index >= pd.to_datetime(start_date)) & (smoothed_polls.index <= pd.to_datetime(end_date))

    # Try to infer the daily DataFrame from smoothed_polls if possible
    parties = ['Con', 'Lab', 'LD', 'Green', 'Ref', 'UKIP', 'BNP', 'TIG/CUK', 'Others']
    parties = [p for p in parties if smoothed_polls.loc[mask,p].max(skipna=True) >= support_threshold]

    n = len(parties)

    # Determine grid layout
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10), sharex=True)
    axes = axes.flatten()
    # Loop over parties and plot in the right subplot
    for i, party in enumerate(parties):
        ax = axes[i]
        # Tight margins
        if tight:
            y_data = smoothed_polls.loc[mask, party]
            y_min = y_data.min()
            y_max = y_data.max()
            margin = tight_margin * (y_max - y_min)
            ax.set_ylim(y_min - margin, y_max + margin)
        
        if party != "Others":
            ax.scatter(daily.index, daily[party], alpha=0.4, s=2.5, label=f'{party} daily mean') # type: ignore
        ax.plot(smoothed_polls.index, smoothed_polls[party], '-', linewidth=1.5, label=f'{party} smoothed')
        ax.set_title(party)
        ax.legend(fontsize=8)
        ax.set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])

        # Event Markets
        for name, d in event_dates.items():
            # Vertical line
            ax.axvline(x=d, color="red", linestyle="--", linewidth=0.5)
            # Label (at the top of the axis)
            ax.text(
                d, ax.get_ylim()[1]*0.95,  # a bit below the top of y-axis
                name,
                rotation=90, ha="right", va="top",
                fontsize=5, color="red"
            )

    # Hide unused subplot slots
    for j in range(len(parties), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Polling Support (Kalman Smoothed)", fontsize=16)
    for ax in axes:
        ax.tick_params(labelbottom=True)   # force x labels
        ax.tick_params(labelleft=True)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')  # rotate all x labels

    plt.tight_layout()
    plt.show()

    # Aggregate Plots
    cols = smoothed_polls.columns[9:18]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=False)
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])
        axes[i].plot(smoothed_polls.index, smoothed_polls[col], linewidth=1.5, label=col)
        axes[i].set_title(col)
        axes[i].legend()
        # Add tight y-limits if requested
        if tight:
            y_data = smoothed_polls.loc[mask, col].dropna()
            y_min, y_max = y_data.min(), y_data.max()
            margin = tight_margin * (y_max - y_min)
            axes[i].set_ylim(y_min - margin, y_max + margin)
        # Event Markers per subplot
        for name, d in event_dates.items():
            axes[i].axvline(x=d, color="red", linestyle="--", linewidth=0.5)
            axes[i].text(
                d, axes[i].get_ylim()[1]*0.95,
                name,
                rotation=90, ha="right", va="top",
                fontsize=5, color="red"
            )

    fig.suptitle("Aggregates", fontsize=16)
    for ax in axes:
        ax.tick_params(labelbottom=True)   # force x labels
        ax.tick_params(labelleft=True)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')  # rotate all x labels

    plt.tight_layout()
    plt.show()

def PlotParties(smoothed_polls: pd.DataFrame,
                start_date=None, end_date=None,
                support_threshold: float = 1.0, tight: bool = False,
                tight_margin: float = 0.05, show_raw: bool = False,
                daily: Optional[pd.DataFrame] = None):
    """
    Plot smoothed lines of major parties on a single chart, with optional raw polls overlay.

    Args:
        smoothed_polls (pd.DataFrame): Smoothed polling data.
        start_date (str or datetime, optional): Start date for plotting.
        end_date (str or datetime, optional): End date for plotting.
        support_threshold (float): Minimum support threshold to include a party.
        tight (bool): If True, use tighter y-limits around the data. Defaults to False.
        tight_margin (float): Proportion margin for tight plotting. Defaults to 0.05.
        show_raw (bool): Whether to overlay raw poll values as scatter.
        daily (pd.DataFrame, optional): Raw daily polling data for scatter.
    """
    if start_date is None:
        start_date = str(smoothed_polls.index.min())
    if end_date is None:
        end_date = str(smoothed_polls.index.max())

    mask = (smoothed_polls.index >= pd.to_datetime(start_date)) & (smoothed_polls.index <= pd.to_datetime(end_date))

    # Event dates
    event_dates = GetEvents(start_date, end_date)

    # Party list and colors
    party_colors = {
        "Con": "blue",
        "Lab": "red",
        "LD": "gold",
        "Green": "green",
        "Ref": "deepskyblue",
        "UKIP": "purple",
        "BNP": "grey",
        "TIG/CUK": "grey",
        "Others": "grey"
    }

    # Filter parties by threshold
    parties = [p for p in party_colors if p in smoothed_polls.columns and smoothed_polls.loc[mask, p].max(skipna=True) >= support_threshold]

    if not parties:
        print("No parties exceed threshold in this period.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for party in parties:
        color = party_colors.get(party, "black")
        if show_raw and daily is not None and party in daily.columns:
            ax.scatter(daily.index, daily[party], alpha=0.3, s=10, color=color, label=f"{party} raw")
        ax.plot(smoothed_polls.index, smoothed_polls[party], linewidth=1.5, color=color, label=f"{party} smoothed")

    # Apply tight y-limits if requested
    if tight:
        y_data = smoothed_polls.loc[mask, parties].values.flatten()
        y_min = np.nanmin(y_data)
        y_max = np.nanmax(y_data)
        margin = tight_margin * (y_max - y_min)
        ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date)) # type: ignore
    ax.set_title("Major Party Polling (Smoothed)")
    ax.set_ylabel("Polling %")
    ax.legend()

    # Event markers
    for name, d in event_dates.items():
        ax.axvline(x=d, color="red", linestyle="--", linewidth=0.5) # type: ignore
        ax.text(
            d, ax.get_ylim()[1]*0.95, # type: ignore
            name,
            rotation=90, ha="right", va="top",
            fontsize=5, color="red"
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    
def PlotPollsDiff(polls_df: pd.DataFrame,
                  diff: int = 1,
                  start_date: Optional[str] = None, end_date: Optional[str] = None,
                  support_threshold: float = 1.0):
    """
    Plot differenced polling time series.

    Args:
      polls_df: DataFrame with datetime index and parties as columns.
      diff: int, differencing lag in days (defaults to daily lag)).
      start_date: str or datetime, first date to include (defaults to first date in timeseries).
      end_date: str or datetime, last date to include (defaults to last date in timeseries).
      support_threshold: float, only plot parties with max support above this threshold in the date range.
    """
    if start_date is None:
        start_date = polls_df.index.min()
    if end_date is None:
        end_date = polls_df.index.max()
    
    # Retrieve dictionary of events
    event_dates = GetEvents(start_date, end_date)  
        
    # Restrict to date range
    sub = polls_df.loc[start_date:end_date].copy()

    # Compute max support for each party in this date range (before differencing)
    max_support = sub.max(skipna=True)
    # Only plot parties above support threshold
    parties_to_plot = [col for col in sub.columns if max_support[col] > support_threshold]

    # Apply differencing if requested
    if diff > 0:
        sub = sub.diff(diff).dropna()

    # Only plot selected parties
    sub = sub[parties_to_plot]

    n = len(parties_to_plot)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows), sharex=True, sharey=False)
    axes = axes.flatten()
    for i, col in enumerate(parties_to_plot):
        ax = axes[i]
        sub[col].plot(ax=ax, linewidth=1.2, marker="o", ms=2)
        if diff > 0:
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(col)
        ax.set_ylabel("Polling Share" if diff == 0 else f"Δ Share (lag={diff})")
        # Add vertical lines and labels for each event
        for name, d in event_dates.items():
            ax.axvline(x=d, color="red", linestyle="--", linewidth=0.5)
            # Place label just below top of y-axis
            ylim = ax.get_ylim()
            ax.text(
                d, ylim[1] * 0.95,
                name,
                rotation=90, ha="right", va="top",
                fontsize=5, color="red"
            )
    # Hide unused axes
    for j in range(len(parties_to_plot), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Polling Trends ({'raw' if diff==0 else f'{diff}-day differences'})", fontsize=16)
    for ax in axes:
        ax.tick_params(labelbottom=True)
        ax.tick_params(labelleft=True)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
def PlotPollsDiffBar(polls_df: pd.DataFrame,
                     diff: int = 1,
                     start_date: Optional[str] = None, end_date: Optional[str] = None,
                     support_threshold: float = 1.0):
    """
    Plot differenced polling time series as bar charts.

    Args:
      polls_df: DataFrame with datetime index and parties as columns.
      diff: int, differencing lag in days (defaults to daily lag).
      start_date: str or datetime, first date to include (defaults to first date in timeseries).
      end_date: str or datetime, last date to include (defaults to last date in timeseries).
      support_threshold: float, only plot parties with max support above this threshold in the date range.
    """
    if start_date is None:
        start_date = polls_df.index.min()
    if end_date is None:
        end_date = polls_df.index.max()
    
    # Retrieve dictionary of events
    event_dates = GetEvents(start_date, end_date)
        
    # Restrict to date range
    sub = polls_df.loc[start_date:end_date].copy()

    # Compute max support for each party in this date range (before differencing)
    max_support = sub.max(skipna=True)
    # Only plot parties above support threshold
    parties_to_plot = [col for col in sub.columns if max_support[col] > support_threshold]

    # Apply differencing if requested
    if diff > 0:
        sub = sub.diff(diff).dropna()

    # Only plot selected parties
    sub = sub[parties_to_plot]

    n = len(parties_to_plot)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows), sharex=True, sharey=False)
    axes = axes.flatten()
    for i, col in enumerate(parties_to_plot):
        ax = axes[i]
        # Plot as bar chart
        ax.bar(sub.index, sub[col], width=0.5, align='center')
        if diff > 0:
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(col)
        ax.set_ylabel("Polling Share" if diff == 0 else f"Δ Share (lag={diff})")
        # Add vertical lines and labels for each event
        for name, d in event_dates.items():
            ax.axvline(x=d, color="red", linestyle="--", linewidth=0.5)
            ylim = ax.get_ylim()
            ax.text(
                d, ylim[1] * 0.95,
                name,
                rotation=90, ha="right", va="top",
                fontsize=5, color="red"
            )
    # Hide unused axes
    for j in range(len(parties_to_plot), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Polling Trends (Bar, {'raw' if diff==0 else f'{diff}-day differences'})", fontsize=16)
    for ax in axes:
        ax.tick_params(labelbottom=True)
        ax.tick_params(labelleft=True)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# ------------------------------------ END -------------------------------------