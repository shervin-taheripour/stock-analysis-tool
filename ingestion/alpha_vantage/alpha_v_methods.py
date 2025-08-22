# alpha_v.py

import os
import pandas as pd
from dotenv import load_dotenv
from alpha_vantage.timeseries import TimeSeries
import requests

def av_get_api_key(env_path: str = "../ingestion/.env") -> str:
    load_dotenv(dotenv_path=env_path, override=True)
    key = os.getenv("ALPHAVANTAGE_KEY")
    if not key:
        raise ValueError(f"ALPHAVANTAGE_KEY not found in {env_path}")
    return key

def av_get_options_chain(
    ticker: str = "SPY",
    env_path: str = "../ingestion/.env",
    expiry: str = None
) -> pd.DataFrame:
    """
    Fetches the options chain for a ticker from Alpha Vantage (REST API).
    Optionally filter for a specific expiry.
    Returns a normalized pandas DataFrame with both calls and puts.
    """
    key = av_get_api_key(env_path)
    url = f"https://www.alphavantage.co/query?function=OPTION_CHAIN&symbol={ticker}&apikey={key}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    if "optionChain" not in data or not data["optionChain"]:
        raise ValueError(f"No options data returned for {ticker}")

    # Each expiry is a dictionary; each contains 'calls' and 'puts' lists
    all_contracts = []
    for exp in data["optionChain"]["result"]:
        exp_date = exp["expirationDate"]
        if expiry and exp_date != expiry:
            continue  # skip if not matching target expiry
        for opt_type in ("calls", "puts"):
            for c in exp[opt_type]:
                all_contracts.append({
                    "ticker": ticker,
                    "expiration_date": exp_date,
                    "option_type": opt_type[:-1],  # 'calls' -> 'call'
                    "contract_symbol": c.get("contractSymbol"),
                    "strike": float(c.get("strike", "nan")),
                    "last_price": float(c.get("lastPrice", "nan")),
                    "bid": float(c.get("bid", "nan")),
                    "ask": float(c.get("ask", "nan")),
                    "volume": int(c.get("volume", 0)),
                    "open_interest": int(c.get("openInterest", 0)),
                    "implied_volatility": float(c.get("impliedVolatility", "nan")),
                    "in_the_money": c.get("inTheMoney"),
                    "last_trade_date": c.get("lastTradeDate"),
                })
    if not all_contracts:
        raise ValueError(f"No contracts found for {ticker} (expiry: {expiry})")
    df = pd.DataFrame(all_contracts)
    # Optional: filter again for a single expiry if you want just one
    if expiry:
        df = df[df["expiration_date"] == expiry]
    return df