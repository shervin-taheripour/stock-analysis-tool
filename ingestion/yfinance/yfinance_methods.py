# yfinance.py

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def yf_get_data(ticker: str = 'AAPL', start: str = None, end: str = None, auto_adjust: bool = True) -> pd.DataFrame:
    """
    Fetch historical OHLCV data using yfinance.

    Parameters
    ----------
    ticker : str
        Stock symbol (e.g., "AAPL")
    start : str or None
        Start date ("YYYY-MM-DD"). Defaults to one year ago.
    end : str or None
        End date ("YYYY-MM-DD"). Defaults to today.
    auto_adjust : bool
        Adjust for splits/dividends if True (default).

    Returns
    -------
    pd.DataFrame
        DataFrame with OHLCV columns and 'Ticker' column.
    """
    if end is None:
        end = datetime.today().date()
    if start is None:
        start = end - timedelta(days=365)

    df = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust)
    df.columns = [col if isinstance(col, str) else col[0] for col in df.columns]
    df["Ticker"] = ticker
    return df

def yf_get_adj_close(ticker: str = 'AAPL', start: str = None, end: str = None) -> pd.DataFrame:
    """
    Fetch adjusted close price using yfinance.

    Returns DataFrame with 'Yahoo_Finance' as column and 'Ticker'.
    """
    df = yf_get_data(ticker, start=start, end=end, auto_adjust=True)

    if 'Adj Close' in df.columns:
        return df[['Adj Close', 'Ticker']].rename(columns={'Adj Close': 'Yahoo_Finance'})
    elif 'Close' in df.columns:
        print("⚠️ Adjusted Close not found — falling back to Close.")
        return df[['Close', 'Ticker']].rename(columns={'Close': 'Yahoo_Finance'})
    else:
        raise KeyError("Neither 'Adj Close' nor 'Close' found in DataFrame columns.")
