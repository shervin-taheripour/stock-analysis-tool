# alpha_v.py

import os
import pandas as pd
from dotenv import load_dotenv
from alpha_vantage.timeseries import TimeSeries

def av_get_api_key(env_path: str = "../ingestion/.env") -> str:
    """
    Loads ALPHAVANTAGE_KEY from the specified .env file.
    """
    load_dotenv(dotenv_path=env_path, override=True)
    key = os.getenv("ALPHAVANTAGE_KEY")
    if not key:
        raise ValueError(f"ALPHAVANTAGE_KEY not found in {env_path}")
    return key

def av_get_daily_history(
    ticker: str = "AAPL",
    env_path: str = "../ingestion/.env",
    output_size: str = "compact"
) -> pd.DataFrame:
    """
    Fetches daily OHLCV data for a ticker from Alpha Vantage.
    Returns a formatted pandas DataFrame.
    output_size: 'compact' (latest 100 points) or 'full' (full history)
    """
    key = av_get_api_key(env_path)
    ts = TimeSeries(key=key, output_format="pandas")
    data, _ = ts.get_daily(symbol=ticker, outputsize=output_size)
    data = data.sort_index()
    data = data.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume"
    })
    data["Ticker"] = ticker
    return data

def av_get_close(
    ticker: str = "AAPL",
    env_path: str = "../ingestion/.env",
    output_size: str = "compact"
) -> pd.DataFrame:
    """
    Fetches only the daily close prices for a ticker from Alpha Vantage.
    """
    key = av_get_api_key(env_path)
    ts = TimeSeries(key=key, output_format="pandas")
    data, _ = ts.get_daily(symbol=ticker, outputsize=output_size)
    data = data.sort_index()
    close = data[["4. close"]].rename(columns={"4. close": "AlphaVantage_Close"})
    close["Ticker"] = ticker
    return close