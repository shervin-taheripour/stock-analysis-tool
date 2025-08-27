import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import utils

st.markdown("""
# 📊 Stock Analysis Tool

Welcome!  
Select a stock from the sidebar to view its company profile and most recent technical indicators.  
All data updates automatically from the project’s research artefacts.

---
""")

# --- Sidebar: Ticker selection ---
meta_kv = utils.load_meta()
tickers = meta_kv['ticker'].unique().tolist()
default_ticker = 'AAPL' if 'AAPL' in tickers else tickers[0]
ticker = st.sidebar.selectbox("Select Ticker", tickers, index=tickers.index(default_ticker))

# --- Load data ---
ohlcv = utils.load_ohlcv()
indicators = utils.load_indicators()

# --- 1. Metadata panel ---
meta_wide = meta_kv.pivot(index="ticker", columns="key", values="value").reset_index()
meta_fields_orig = ["longName", "sector", "country", "currency", "latest_price"]
meta_fields_disp = ["name", "sector", "country", "currency", "price"]

meta_row = meta_wide[meta_wide["ticker"] == ticker]
meta_dict = {}
for orig, disp in zip(meta_fields_orig, meta_fields_disp):
    if orig == "latest_price":
        price = utils.get_latest_price(ohlcv, ticker)
        if isinstance(price, (float, int)) and pd.notna(price):
            meta_dict[disp] = round(price, 2)
        else:
            meta_dict[disp] = price
    elif orig in meta_row.columns:
        val = meta_row[orig].values[0]
        meta_dict[disp] = val if pd.notna(val) else "not available"
    else:
        meta_dict[disp] = "not available"

meta_df = pd.DataFrame([meta_dict], columns=meta_fields_disp)
st.markdown("### Company Metadata")
st.dataframe(meta_df.style.hide(axis="index"), use_container_width=True)

# --- 2. Latest indicators panel ---
ind_fields = ["sma20", "sma50", "rsi14", "vol20", "vol60"]
ind = indicators[indicators["ticker"] == ticker].sort_values("date").tail(1)
ind_dict = {field: (ind[field].values[0] if field in ind and pd.notna(ind[field].values[0]) else "not available") for field in ind_fields}
ind_df = pd.DataFrame([ind_dict], columns=ind_fields)
st.markdown("### Latest Technical Indicators")
st.dataframe(ind_df.style.hide(axis="index"), use_container_width=True)

st.markdown("### Technical Indicator Panel")
try:
    fig, axs = utils.plot_indicators_improved(ohlcv, ticker, lookback=252)
    st.pyplot(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Could not render indicator panel for {ticker}: {e}")

st.markdown("""
**Technical Indicator Panel**

Displays four classic trading indicators for the selected stock:
- **Bollinger Bands**: Shows typical price range; price near the band edges may signal overbought/oversold.
- **RSI (Relative Strength Index)**: Momentum gauge; values above 70 = overbought, below 30 = oversold.
- **MACD**: Trend and momentum; look for crossovers or histogram color shifts.
- **Volume**: Tracks daily trading activity.
""")

st.markdown("### Candlestick Chart")
try:
    fig, axes = utils.plot_candlestick(ohlcv, ticker, lookback=60)
    st.pyplot(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Could not render candlestick chart for {ticker}: {e}")

# must create the `returns` DataFrame from ohlcv for this dashboard
# (e.g., daily log or percent returns for each ticker)

@st.cache_data
def load_returns(ohlcv):
    return (
        ohlcv.sort_values("date")
        .pivot(index="date", columns="ticker", values="close")
        .pct_change()
        .dropna(how="all")
    )

returns = load_returns(ohlcv)

st.markdown("""
**Candlestick & Volume Chart**

Visualizes daily Open, High, Low, and Close prices ("OHLCV") for the selected stock, with volume bars below.
Candlesticks make it easy to see price moves and volatility. The moving average line (blue) shows recent trend.
""")

# Compute CAPM table (cache for speed)
capm_df = utils.compute_capm_betas(returns, market_col="SPY")

# Ticker selection (already from sidebar)
asset = ticker  # current selected asset (from sidebar)
market_col = "SPY"  # can let user pick later

st.markdown("### Portfolio Return Analytics")
try:
    fig, axs = utils.plot_portfolio_returns_panel(returns, capm_df, asset=asset, market_col=market_col)
    st.pyplot(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Could not render portfolio returns panel for {asset}: {e}")

st.markdown("""
**Portfolio Return Analytics**

- **Left plot:** Scatter of the selected stock's daily return vs. the S&P 500 (SPY), with the red line showing the CAPM fit.
  - **Beta:** Slope—how sensitive this stock is to market moves. Beta > 1 = moves more than market, < 1 = less.
  - **Alpha:** Intercept—measures outperformance vs. market.
- **Right plot:** Growth of $1 for three portfolios:
  - The selected stock alone (orange dashed)
  - An equally-weighted basket of all tickers (blue)
  - A volatility-weighted basket (green)
""")

st.header("Option Analytics")

# --- Greeks Table ---
st.subheader("Greeks (Δ, Γ, Vega, Theta, Rho)")
greeks = utils.get_latest_greeks(utils.load_delta_hedge(), ticker)
if not greeks.empty:
    st.dataframe(greeks)
else:
    st.info("No greeks found for this ticker.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Delta Hedge P&L")
    fig = utils.plot_delta_hedge_pnl(utils.load_delta_hedge(), ticker)
    if fig: st.pyplot(fig)
    else: st.info("No P&L data for this ticker.")

with col2:
    st.subheader("Binomial Prices (Strikes)")
    fig = utils.plot_binomial_vs_strike(utils.load_binomial_prices(), ticker)
    if fig: st.pyplot(fig)
    else: st.info("No binomial price data for this ticker.")

col3, col4 = st.columns(2)
with col3:
    st.subheader("Monte Carlo Simulated Paths")
    mc_gbm = utils.load_mc_gbm_paths()
    fig1, _ = utils.plot_mc_gbm_panel(mc_gbm, ticker)
    if fig1: st.pyplot(fig1)
    else: st.info("No MC paths for this ticker.")

with col4:
    st.subheader("GBM Terminal Price Distribution")
    mc_gbm = utils.load_mc_gbm_paths()
    _, fig2 = utils.plot_mc_gbm_panel(mc_gbm, ticker)
    if fig2: st.pyplot(fig2)
    else: st.info("No MC histogram for this ticker.")

st.markdown("""
**Options Analytics Panel**

- **Delta Hedge P&L:** Shows profit/loss of dynamically delta-hedging an ATM call. Visualizes risk reduction by rebalancing as price moves.
- **Binomial Option Prices vs Strike:** European/American option prices for a range of strikes. See how early exercise and moneyness affect values.
- **Monte Carlo GBM Paths:** Simulated price paths using Geometric Brownian Motion (risk-neutral). Each “spaghetti” line is a possible outcome.
- **Terminal Price Histogram:** Histogram of end prices from Monte Carlo, showing the distribution at expiry.

---

**Implied Volatility Table (Newton Method)**

Shows implied volatility for each contract, extracted by inverting Black-Scholes via Newton's method. Implied vol reflects the market’s consensus on future volatility.
""")

st.subheader("Monte Carlo Option Price Table")
mc_option = utils.get_mc_option_summary(utils.load_mc_option_prices(), ticker)
if not mc_option.empty:
    st.dataframe(mc_option)
else:
    st.info("No MC summary for this ticker.")
st.markdown("""
    *Shows simulated option prices (and standard errors) from 30,000 Monte Carlo paths (GBM model), 
    compared to Black-Scholes closed-form. Useful for checking numerical convergence and pricing robustness.*
    """)
st.subheader("Implied Volatility Table (Newton method)")
iv_table = utils.get_iv_table(utils.load_iv_universe(), ticker)
if not iv_table.empty:
    st.dataframe(iv_table)
else:
    st.info("No IV data for this ticker.")
st.markdown("""
    *Implied volatility is the market's consensus forecast of future volatility, 
    extracted by inverting the Black-Scholes formula with Newton's method for each option contract.
    """)

