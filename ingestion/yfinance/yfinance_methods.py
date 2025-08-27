# yfinance.py

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def get_ohlcv_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(start=start, end=end).copy()
    df.reset_index(inplace=True)
    df["ticker"] = ticker
    df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Dividends": "dividends",
        "Stock Splits": "splits"
    }, inplace=True)
    return df[["ticker", "date", "open", "high", "low", "close", "volume", "dividends", "splits"]]

def get_ohlcv_data_db(ticker: str, ticker_id: int, start: str, end: str) -> pd.DataFrame:
    df = get_ohlcv_data(ticker, start, end)
    if 'ticker' in df.columns:
        df.drop('ticker', axis=1, inplace=True)
    df["ticker_id"] = ticker_id
    return df[["ticker_id", "date", "open", "high", "low", "close", "volume", "dividends", "splits"]]


def get_company_metadata(ticker: str) -> pd.DataFrame:
    info = yf.Ticker(ticker).info
    df = pd.DataFrame(info.items(), columns=["key", "value"])
    df["ticker"] = ticker
    return df[["ticker", "key", "value"]]

def get_company_metadata_db(ticker: str, ticker_id: int) -> pd.DataFrame:
    df = get_company_metadata(ticker)
    if 'ticker' in df.columns:
        df.drop('ticker', axis=1, inplace=True)
    df["ticker_id"] = ticker_id
    return df[["ticker_id", "key", "value"]]


def get_options_chain(ticker: str, expiration: str) -> pd.DataFrame:
    chain = yf.Ticker(ticker).option_chain(expiration)
    df_calls = chain.calls.copy()
    df_calls["option_type"] = "call"
    df_puts = chain.puts.copy()
    df_puts["option_type"] = "put"
    df = pd.concat([df_calls, df_puts], ignore_index=True)
    df["ticker"] = ticker
    df["expiration_date"] = expiration
    df.rename(columns={
        "contractSymbol": "contract_symbol",
        "lastTradeDate": "last_trade_date",
        "lastPrice": "last_price",
        "percentChange": "percent_change",
        "openInterest": "open_interest",
        "impliedVolatility": "implied_volatility",
        "inTheMoney": "in_the_money",
        "contractSize": "contract_size"
    }, inplace=True)
    return df

def get_options_chain_db(ticker: str, ticker_id: int, expiration: str) -> pd.DataFrame:
    df = get_options_chain(ticker, expiration)
    if 'ticker' in df.columns:
        df.drop('ticker', axis=1, inplace=True)
    df["ticker_id"] = ticker_id
    return df[["ticker_id", "contract_symbol", "last_trade_date", "strike", "last_price", "bid", "ask", "change", "percent_change", "volume", "open_interest", "implied_volatility", "in_the_money", "contract_size", "currency", "option_type", "expiration_date"]]


def get_financials_long(ticker: str, kind: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    if kind == "income":
        df = tk.financials
    elif kind == "balance":
        df = tk.balance_sheet
    elif kind == "cashflow":
        df = tk.cashflow
    else:
        return pd.DataFrame()
    df = df.reset_index().melt(id_vars="index", var_name="report_date", value_name="value")
    df.columns = ["metric_name", "report_date", "value"]
    df["ticker"] = ticker
    return df[["ticker", "report_date", "metric_name", "value"]]

def get_events_calendar(ticker: str) -> pd.DataFrame:
    calendar = yf.Ticker(ticker).calendar
    if not isinstance(calendar, dict) and calendar is not None:
        calendar = calendar.to_dict()
    if not calendar:
        return pd.DataFrame()
    df = pd.DataFrame(calendar.items(), columns=["event_type", "event_date"])
    df["ticker"] = ticker
    return df[["ticker", "event_type", "event_date"]]


def get_sustainability_data(ticker: str) -> pd.DataFrame:
    sustain = yf.Ticker(ticker).sustainability
    if sustain is None:
        return pd.DataFrame()
    df = sustain.reset_index()
    df.columns = ["metric_name", "esg_score"]
    df["ticker"] = ticker
    return df[["ticker", "metric_name", "esg_score"]]


def get_dividends(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).dividends.reset_index()
    df.columns = ["date", "dividends"]
    df["ticker"] = ticker
    return df[["ticker", "date", "dividends"]]


def get_splits(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).splits.reset_index()
    df.columns = ["date", "splits"]
    df["ticker"] = ticker
    return df[["ticker", "date", "splits"]]


def get_earnings_dates(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).earnings_dates
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.columns = ["earnings_date", "eps_estimate", "reported_eps", "surprise"]
    df["ticker"] = ticker
    return df[["ticker", "earnings_date", "eps_estimate", "reported_eps", "surprise"]]

def get_analyst_recommendations(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).recommendations
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={
        "period": "period",
        "strongBuy": "strong_buy",
        "buy": "buy",
        "hold": "hold",
        "sell": "sell",
        "strongSell": "strong_sell"
    })
    df["ticker"] = ticker
    return df[["ticker", "period", "strong_buy", "buy", "hold", "sell", "strong_sell"]]

