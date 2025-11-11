import pandas as pd
import numpy as np
import statsmodels.api as sm

# ------------------------------ Smoothing & Aggs ------------------------------
# Two pass smoothing
def SmoothPolls(daily: pd.DataFrame):
    """Smooths daily polls by two pass approach using Kalman Filter.

    Args:
        daily (pd.DataFrame): Aggregated and cleaned daily polls

    Returns:
        smoothed_polls (pd.DataFrame): _description_
    """
    daily.index = pd.to_datetime(daily.index) # type: ignore
    daily = daily.sort_index()

    smoothed_first_pass = pd.DataFrame(index=daily.index, columns=daily.
                                       columns, dtype=float)
    smoothed_polls = pd.DataFrame(index=daily.index, columns=daily.columns, 
                                  dtype=float)

    # First pass: smooth between observed polls
    for party in daily.columns:
        series = daily[party].asfreq('D')
        if series.notna().sum() < 2:
            continue

        mod = sm.tsa.UnobservedComponents(series, level='local level') # type: ignore
        res = mod.fit(disp=False)
        
        smoothed_series = res.smoothed_state[0] # type: ignore
        
        # Keep NaNs if no observations
        smoothed_series[series.isna()] = np.nan
        smoothed_first_pass[party] = smoothed_series

    # Second pass: smooth outside observations
    for party in daily.columns:
        series = smoothed_first_pass[party].copy()
        
        first_valid = series.first_valid_index()
        last_valid = series.last_valid_index()
        if first_valid is None:
            continue
        
        # Fill leading/trailing NaNs with 0
        series[:first_valid] = 0
        series[last_valid + pd.Timedelta(days=1):] = 0 # type: ignore
        
        # Smooth full series
        series = series.asfreq('D')
        mod = sm.tsa.UnobservedComponents(series, level='local level') # type: ignore
        res = mod.fit(disp=False)
        
        smoothed_polls[party] = res.smoothed_state[0] # type: ignore
    
    return smoothed_polls

# Produce aggregates
def AggPolls(smoothed_polls: pd.DataFrame):
    """Processes smoothed polls creating aggregates.

    Args:
        smoothed_polls (pd.DataFrame): Smoothed polls

    Returns:
        smoothed_polls (pd.DataFrame): Smoothed polls with aggregates
    """
    # Calculate support for non major parties
    smoothed_polls['Others'] = 100 - smoothed_polls.sum(axis = 1)
    # Aggregates
    smoothed_polls['Right'] = smoothed_polls['Con'] + smoothed_polls['Ref'] + smoothed_polls['UKIP'] + smoothed_polls['BNP']
    smoothed_polls['Left'] = smoothed_polls['Lab'] + smoothed_polls['LD'] + smoothed_polls['Green'] + smoothed_polls['TIG/CUK']
    smoothed_polls['FarRight'] = smoothed_polls['Ref'] + smoothed_polls['UKIP'] + smoothed_polls['BNP']
    smoothed_polls['NonLabLeft'] = smoothed_polls['Left'] - smoothed_polls['Lab']
    smoothed_polls['LAB/Left'] = 100 * smoothed_polls['Lab'] / smoothed_polls['Left']
    smoothed_polls['CON/Right'] = 100 * smoothed_polls['Con'] / smoothed_polls['Right']
    smoothed_polls['FarRight/Right'] = 100 * smoothed_polls['FarRight'] / smoothed_polls['Right']
    smoothed_polls['Major'] = smoothed_polls['Con'] + smoothed_polls['Lab']
    smoothed_polls['Minor'] = 100 - smoothed_polls['Major']
    smoothed_polls['Others_exp'] = smoothed_polls['Others'] + smoothed_polls['TIG/CUK']
    smoothed_polls['LeftPlus'] = smoothed_polls['Left'] + smoothed_polls['Others']
    smoothed_polls['NonLabLeftPlus'] = smoothed_polls['NonLabLeft'] + smoothed_polls['Others']
    smoothed_polls['Lab/LeftPlus'] = 100 * smoothed_polls['Lab'] / smoothed_polls['LeftPlus']
    
    return smoothed_polls

# ------------------------------------ END -------------------------------------