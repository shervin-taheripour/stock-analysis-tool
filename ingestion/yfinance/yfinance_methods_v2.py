# =========================
# FILE: yfinance_methods_v2.py
# =========================
# Lightweight, no‑nonsense helpers that match your original functions
# and add small robustness (optional period / start-end, UTC dates,
# consistent column names, empty‑DataFrame fallbacks).
#
# Keep it simple: one file you can import in your EDA notebook.
# Functions mirror your list, not over‑engineered.
# Safe when yfinance returns empty/None.

from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional

# ----------------------
# Small internal helpers
# ----------------------

def _empty(cols: List[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def _ensure_dt(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def _history(ticker: str,
             start: Optional[str] = None,
             end: Optional[str] = None,
             period: Optional[str] = "max",
             interval: str = "1d",
             auto_adjust: bool = False,
             prepost: bool = False,
             actions: bool = True) -> pd.DataFrame:
    """Wrapper that accepts either (period) or (start/end)."""
    t = yf.Ticker(ticker)
    kwargs = dict(interval=interval, auto_adjust=auto_adjust, prepost=prepost, actions=actions)
    if period is not None:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        kwargs["end"] = end
    try:
        out = t.history(**kwargs)
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ----------------------
# Public API (tables)
# ----------------------

def get_ohlcv_data(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = "max",
    interval: str = "1d",
    auto_adjust: bool = False,
    prepost: bool = False,
) -> pd.DataFrame:
    """
    OHLCV (+ adj_close, dividends, splits from yfinance history).
    Returns columns:
    ['ticker','date','open','high','low','close','adj_close','volume','dividends','splits']
    """
    df = _history(ticker, start, end, period, interval, auto_adjust, prepost, actions=True)
    if df.empty:
        return _empty(["ticker","date","open","high","low","close","adj_close","volume","dividends","splits"])  # type: ignore

    df = df.reset_index().rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
        "Dividends": "dividends",
        "Stock Splits": "splits",
    })
    df["ticker"] = ticker
    df = _ensure_dt(df, "date")
    # dtypes (best effort, non‑fatal if missing)
    for c in ["open","high","low","close","adj_close","dividends","splits"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" in df: df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")

    keep = ["ticker","date","open","high","low","close","adj_close","volume","dividends","splits"]
    for k in keep:
        if k not in df: df[k] = pd.NA
    return df[keep]


def get_company_metadata(ticker: str) -> pd.DataFrame:
    """Key/value dump of yf.Ticker(...).get_info()/info (simple + stable)."""
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.get_info() if hasattr(t, "get_info") else t.info
    except Exception:
        info = {}
    if not info:
        return _empty(["ticker","key","value"])  # type: ignore
    df = pd.DataFrame(info.items(), columns=["key","value"])  # type: ignore
    df["ticker"] = ticker
    return df[["ticker","key","value"]]


def get_options_chain(ticker: str, expiration: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    try:
        ch = t.option_chain(expiration)
    except Exception:
        return _empty(["ticker","expiration_date","option_type","contract_symbol"])  # minimal cols
    calls = getattr(ch, "calls", pd.DataFrame()).copy()
    puts = getattr(ch, "puts", pd.DataFrame()).copy()
    if calls.empty and puts.empty:
        return _empty(["ticker","expiration_date","option_type","contract_symbol"])  # type: ignore
    calls["option_type"] = "call"
    puts["option_type"] = "put"
    df = pd.concat([calls, puts], ignore_index=True)
    df["ticker"] = ticker
    df["expiration_date"] = expiration

    # Rename columns to standard names
    df = df.rename(columns={
        "contractSymbol": "contract_symbol",
        "lastTradeDate": "last_trade_date",
        "lastPrice": "last_price",
        "percentChange": "percent_change",
        "openInterest": "open_interest",
        "impliedVolatility": "implied_volatility",
        "inTheMoney": "in_the_money",
        "contractSize": "contract_size",
    })

    # Ensure last_trade_date is timezone-naive datetime
    df = _ensure_dt(df, "last_trade_date")
    
    # Ensure expiration_date is timezone-naive datetime
    df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce")
    df["expiration_date"] = df["expiration_date"].dt.tz_localize(None)

    return df


def get_financials_long(ticker: str, kind: str) -> pd.DataFrame:
    """
    Long format fundamentals.
    kind ∈ {"income","balance","cashflow","q_income","q_balance","q_cashflow"}
    """
    t = yf.Ticker(ticker)
    src = None
    try:
        if kind == "income":
            src = t.financials
        elif kind == "balance":
            src = t.balance_sheet
        elif kind == "cashflow":
            src = t.cashflow
        elif kind == "q_income":
            src = t.quarterly_financials
        elif kind == "q_balance":
            src = t.quarterly_balance_sheet
        elif kind == "q_cashflow":
            src = t.quarterly_cashflow
        else:
            return _empty(["ticker","report_date","metric_name","value"])  # type: ignore
    except Exception:
        src = None
    if src is None or src.empty:
        return _empty(["ticker","report_date","metric_name","value"])  # type: ignore

    df = src.reset_index().melt(id_vars=src.index.name or "index",
                                var_name="report_date",
                                value_name="value")
    df.columns = ["metric_name","report_date","value"]
    df["ticker"] = ticker
    df = _ensure_dt(df, "report_date")
    return df[["ticker","report_date","metric_name","value"]]


def get_events_calendar(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    try:
        cal = t.calendar
    except Exception:
        cal = None
    if cal is None:
        return _empty(["ticker","event_type","event_date"])  # type: ignore
    # yfinance sometimes returns DataFrame, sometimes Series/dict
    if isinstance(cal, pd.DataFrame) or isinstance(cal, pd.Series):
        cal = cal.to_dict()
    if not isinstance(cal, dict) or not cal:
        return _empty(["ticker","event_type","event_date"])  # type: ignore
    df = pd.DataFrame(cal.items(), columns=["event_type","event_date"])  # type: ignore
    df["ticker"] = ticker
    df = _ensure_dt(df, "event_date")
    return df[["ticker","event_type","event_date"]]


def get_sustainability_data(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    try:
        sus = t.sustainability
    except Exception:
        sus = None
    if sus is None or (hasattr(sus, "empty") and sus.empty):
        return _empty(["ticker","metric_name","esg_score"])  # type: ignore
    df = sus.reset_index().copy()
    df.columns = ["metric_name","esg_score"]
    df["ticker"] = ticker
    return df[["ticker","metric_name","esg_score"]]


def get_dividends(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    try:
        df = t.dividends.reset_index()
    except Exception:
        return _empty(["ticker","date","dividends"])  # type: ignore
    if df.empty:
        return _empty(["ticker","date","dividends"])  # type: ignore
    df.columns = ["date","dividends"]
    df["ticker"] = ticker
    df = _ensure_dt(df, "date")
    df["dividends"] = pd.to_numeric(df["dividends"], errors="coerce")
    return df[["ticker","date","dividends"]]


def get_splits(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    try:
        df = t.splits.reset_index()
    except Exception:
        return _empty(["ticker","date","splits"])  # type: ignore
    if df.empty:
        return _empty(["ticker","date","splits"])  # type: ignore
    df.columns = ["date","splits"]
    df["ticker"] = ticker
    df = _ensure_dt(df, "date")
    df["splits"] = pd.to_numeric(df["splits"], errors="coerce")
    return df[["ticker","date","splits"]]


def get_earnings_dates(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    try:
        df = t.earnings_dates
    except Exception:
        return _empty(["ticker","earnings_date","eps_estimate","reported_eps","surprise"])  # type: ignore
    if df is None or df.empty:
        return _empty(["ticker","earnings_date","eps_estimate","reported_eps","surprise"])  # type: ignore
    df = df.reset_index()
    # yfinance may return columns with spaces; normalize
    rename = {
        df.columns[0]: "earnings_date",
    }
    if "EPS Estimate" in df.columns: rename["EPS Estimate"] = "eps_estimate"
    if "Reported EPS" in df.columns: rename["Reported EPS"] = "reported_eps"
    if "Surprise(%)" in df.columns: rename["Surprise(%)"] = "surprise"
    df = df.rename(columns=rename)
    for c in ["eps_estimate","reported_eps","surprise"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ticker"] = ticker
    df = _ensure_dt(df, "earnings_date")
    keep = ["ticker","earnings_date","eps_estimate","reported_eps","surprise"]
    for k in keep:
        if k not in df: df[k] = pd.NA
    return df[keep]


def get_analyst_recommendations_summary(ticker: str) -> pd.DataFrame:
    """If available in your yfinance version (period/strongBuy/buy/...), return summary."""
    t = yf.Ticker(ticker)
    try:
        df = t.recommendations_summary
    except Exception:
        df = None
    if df is None or df.empty:
        return _empty(["ticker","period","strong_buy","buy","hold","sell","strong_sell"])  # type: ignore
    df = df.rename(columns={
        "period": "period",
        "strongBuy": "strong_buy",
        "buy": "buy",
        "hold": "hold",
        "sell": "sell",
        "strongSell": "strong_sell",
    })
    df["ticker"] = ticker
    return df[["ticker","period","strong_buy","buy","hold","sell","strong_sell"]]


def get_analyst_recommendations_history(ticker: str) -> pd.DataFrame:
    """Traditional recommendations table: firm / toGrade / fromGrade / action over time."""
    t = yf.Ticker(ticker)
    try:
        df = t.recommendations
    except Exception:
        return _empty(["ticker","date","firm","to_grade","from_grade","action"])  # type: ignore
    if df is None or df.empty:
        return _empty(["ticker","date","firm","to_grade","from_grade","action"])  # type: ignore
    df = df.reset_index().rename(columns={
        df.columns[0]: "date",
        "Firm": "firm",
        "To Grade": "to_grade",
        "From Grade": "from_grade",
        "Action": "action",
    })
    df["ticker"] = ticker
    df = _ensure_dt(df, "date")
    return df[["ticker","date","firm","to_grade","from_grade","action"]]


# Backwards‑compatible alias matching your earlier naming
get_analyst_recommendations = get_analyst_recommendations_summary


# ----------------------
# Simple multi‑ticker helper (optional)
# ----------------------

def collect_for_universe(tickers: List[str],
                         start: Optional[str] = None,
                         end: Optional[str] = None,
                         period: Optional[str] = "max",
                         interval: str = "1d",
                         auto_adjust: bool = False,
                         prepost: bool = False) -> Dict[str, pd.DataFrame]:
    """One‑shot collector you can call from a notebook. Returns dict of concatenated tables."""
    buckets = {
        "ohlcv": [],
        "dividends": [],
        "splits": [],
        "earnings_dates": [],
        "meta_kv": [],
    }
    for tk in tickers:
        buckets["ohlcv"].append(get_ohlcv_data(tk, start, end, period, interval, auto_adjust, prepost))
        buckets["dividends"].append(get_dividends(tk))
        buckets["splits"].append(get_splits(tk))
        buckets["earnings_dates"].append(get_earnings_dates(tk))
        buckets["meta_kv"].append(get_company_metadata(tk))
    out = {k: (pd.concat(v, ignore_index=True) if v else _empty([])) for k, v in buckets.items()}
    return out


# ==================================
# FILE: ingest_master.py  (optional)
# ==================================
# Tiny script to run from the terminal to materialize parquet files for EDA/DB.
# If preferred *not* to use a separate script, just call collect_for_universe()
# inside notebook and save the returned DataFrames there.

if __name__ == "__main__":
    # Example universe (20‑ticker list)
    TICKERS = [
        "AAPL","MSFT","NVDA","AMZN","META","GOOG","TSLA",
        "JPM","XOM","BRK-B","UNH","SPY","QQQ","DIA","IWM","XLK","XLF","XLV","XLE","TLT",
    ]
    dfs = collect_for_universe(TICKERS, period="max", interval="1d")

    from pathlib import Path
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs["ohlcv"].to_parquet(out_dir / "ohlcv.parquet", index=False)
    dfs["dividends"].to_parquet(out_dir / "dividends.parquet", index=False)
    dfs["splits"].to_parquet(out_dir / "splits.parquet", index=False)
    dfs["earnings_dates"].to_parquet(out_dir / "earnings_dates.parquet", index=False)
    dfs["meta_kv"].to_parquet(out_dir / "meta_kv.parquet", index=False)

    print("Saved parquet files to", out_dir.resolve())
