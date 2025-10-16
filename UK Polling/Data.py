import pandas as pd
import numpy as np
import re
import urllib.request

# ------------------------- Data Retrieval & Cleaning --------------------------
# Utilities to retrieve latest polls
def ParsePollDates(s: str | float | None,
                    default_year: int = 2025
                    ) -> pd.Timestamp | None:
    """Parse polling dates from Wikipedia page for use in function for retrieving latest polling data.
    
    Returns first day of range as 'YYYY/MM/DD'.

    Args:
        s (str | float | None): _description_
        default_year (int, optional): _description_. Defaults to 2025.

    Returns:
        pd.Timestamp | None: _description_
    """
    if pd.isna(s):
        return None
    
    s = str(s).strip()

    s = s.replace("\xa0", " ").strip()
    
    s = re.sub(r"[–—−]", "-", s)
    
    s = re.sub(r"\s*-\s*", "-", s)
    
    parts = s.split("-")
    first_part = parts[0].strip()
    
    if not re.search(r"[A-Za-z]", first_part) and len(parts) > 1:
        month_match = re.search(r"[A-Za-z]+", parts[1])
        month = month_match.group(0) if month_match else ""
        first_part = f"{first_part} {month}"
    
    first_part = re.sub(r"\s+", " ", first_part)
    
    date_str = f"{first_part} {default_year}"
    
    try:
        dt = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
        return None if pd.isna(dt) else dt
    except Exception:
        return None

def GetLatestPolls(polls):
    """Returns lates polling data that is not included on the database.

    Args:
        polls (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    url = "https://en.wikipedia.org/wiki/Opinion_polling_for_the_next_United_Kingdom_general_election#2025"
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    html = urllib.request.urlopen(req).read()

    tables = pd.read_html(html, flavor="lxml")

    # Identify the 2025 table
    polls_new = None
    for t in tables:
        if "Fieldwork date(s)" in t.columns or "Dates conducted" in t.columns:
            polls_new = t
            break

    if polls_new is None:
        raise ValueError("Could not find 2025 polling table")

    # Match party names to df
    polls_new = polls_new.rename(columns={
        "Fieldwork date(s)": "dates_conducted",
        "Polling firm/Commissioner": "pollster",
        "Sample size": "sample_size",
        "Con": "Con",
        "Lab": "Lab",
        "Lib Dem": "LD",
        "Green": "Grn",
        "Reform UK": "Ref",
        "SNP": "SNP",
        "Plaid": "PC",
        "Others": "Others",
        "Lead": "Lead"
    })
    
    # Drop irrelevant entries & relabel
    polls_new = polls_new.drop(columns=['Pollster','Client','Area','sample_size','Others','Lead','SNP','PC'])
    polls_new = polls_new.rename(columns={'Grn':'Green','Dates conducted':'Date'})
    
    # Reformat column headers
    polls_new.columns = polls_new.columns.get_level_values(0)
    
    # Remove redundant info
    parties = ['Con','Lab','Ref','LD','Green']
    polls_new[parties] = polls_new[parties].replace('%','', regex=True)
    polls_new[parties] = polls_new[parties].apply(pd.to_numeric, errors="coerce")
    polls_new = polls_new.dropna(subset=parties)
    polls_new = polls_new.reset_index(drop=True)
    
    # Match date format
    polls_new['Date'] = polls_new['Date'].apply(ParsePollDates)
    
    # Drop last entry (poll spans years 2024-25)
    polls_new = polls_new.iloc[:-1].reset_index(drop=True)
    
    # Drop polls already included
    latest_date = polls['Date'].max()
    
    # Filter new_polls to include only dates after latest_date
    polls['Date'] = pd.to_datetime(polls['Date'], format="%Y/%m/%d", errors='coerce')
    polls_new['Date'] = pd.to_datetime(polls_new['Date'], format="%Y/%m/%d", errors='coerce')
    latest_date = polls['Date'].max()
    polls_new = polls_new[polls_new['Date'] > latest_date].reset_index(drop=True)
    polls_new['Date'] = polls_new['Date'].dt.strftime("%Y/%m/%d")
    
    # Reorder to match polls
    target_cols = ["Date", "Con", "Lab", "LD", "Green", "Ref", "UKIP", "BNP", "TIG/CUK"]
    for col in target_cols:
        if col not in polls_new.columns:
            polls_new[col] = np.NaN
    polls_new = polls_new[target_cols]
    
    # Convert date format
    polls_new['Date'] = pd.to_datetime(polls_new['Date'], errors='coerce')

    # Reorders columns & add missing parties
    target_cols = ["Date", "Con", "Lab", "LD", "Green", "Ref", "UKIP", "BNP", "TIG/CUK"]
    for col in target_cols:
        if col not in polls_new.columns:
            polls_new[col] = 0.0

    polls_new = polls_new[target_cols]
    
    return polls_new

# Read, clean & process data
def GetPolls():
    """Imports polls from historic csv file, latest polls post-June 2025. Cleans and aggregates into format for smooothing. 

    Returns:
        daily (pd.DataFrame): Polls cleaned and aggregated for smoothing.
    """
    # Load raw historic polling data 1992 - Jun 2025
    polls = pd.read_csv('/Users/constantineparapoulis/Documents/Projects/UK Polling Stuff/Polling Data/Polls.csv')

    # Harmonise data formats
    polls['Date'] = pd.to_datetime(polls['Date'], format='%d/%m/%Y', dayfirst=True, errors='coerce')
    polls.columns = [str(col) for col in polls.columns]

    # Merge with latest polls - scraped off wikipedia; June 2025 - 
    polls_new = GetLatestPolls(polls)
    polls = pd.concat([polls, polls_new], ignore_index=True)
    polls['Date'] = pd.to_datetime(polls['Date'], dayfirst=True, errors='coerce')

    # Incorporate general election results
    ge_df = pd.read_csv('/Users/constantineparapoulis/Documents/Projects/UK Polling Stuff/Polling Data/GE.csv', index_col=0)

    # Cleaning & processing
    ge_df.index = pd.to_datetime(ge_df.index, format='%d/%m/%Y')
    ge_df = ge_df.reset_index().rename(columns={"index": "Date"})
    polls['Type'] = 'Poll'
    ge_df['Type'] = 'GE'
    polls = pd.concat([polls, ge_df],ignore_index=True)
    polls = polls.sort_values('Date').reset_index(drop=True)

    # Grouping & resampling - multiple entries per day
    daily = polls.drop(columns='Type')
    daily = daily.groupby('Date').mean()
    daily = daily.resample('D').mean()
    
    return daily

# ------------------------------------- END ------------------------------------