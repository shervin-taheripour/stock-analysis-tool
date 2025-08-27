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

# ----------------------------------------------------------------------
# Method: Write daily OHLCV DataFrame to Postgres
# ----------------------------------------------------------------------

from sqlalchemy import create_engine

def av_write_daily_to_db(
    df: pd.DataFrame,
    env_path: str = "../ingestion/.env",
    table_name: str = "alpha_vantage_ohlcv"
) -> None:
    """
    Writes Alpha Vantage daily OHLCV DataFrame to a Postgres table.
    DB_URL must be defined in the provided .env file.

    Parameters:
    - df: DataFrame returned by av_get_daily_history()
    - env_path: Path to the .env file with DB credentials
    - table_name: Name of the table to write to (default: alpha_vantage_ohlcv)
    """
    load_dotenv(dotenv_path=env_path, override=True)
    db_url = os.getenv("DB_URL")
    if not db_url:
        raise ValueError("DB_URL not found in .env file")

    df_to_write = df.copy()
    df_to_write.reset_index(inplace=True)  # Ensure 'date' is a column

    engine = create_engine(db_url)
    df_to_write.to_sql(table_name, engine, if_exists="append", index=False)
    print(f"✅ Wrote {len(df_to_write)} rows to table '{table_name}'.")
