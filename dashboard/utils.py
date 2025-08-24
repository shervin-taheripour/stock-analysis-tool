# dashboard/utils.py

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.ticker as mticker
from typing import Optional, Tuple
import statsmodels.api as sm

DATA_DIR = Path(__file__).parent.parent / "analysis" / "data"

# --- Data Loaders ---

def load_ohlcv():
    return pd.read_csv(DATA_DIR / "ohlcv.csv", parse_dates=["date"])

def load_indicators():
    return pd.read_csv(DATA_DIR / "indicators.csv", parse_dates=["date"])

def load_meta():
    return pd.read_csv(DATA_DIR / "meta_kv.csv")

def load_delta_hedge():
    return pd.read_csv(DATA_DIR / "delta_hedge_full_universe_30d.csv", parse_dates=["date"])

def load_binomial_prices():
    return pd.read_csv(DATA_DIR / "binomial_prices_full_universe.csv")

def load_mc_option_prices():
    return pd.read_csv(DATA_DIR / "mc_option_prices_full_universe.csv")

def load_mc_gbm_paths():
    return pd.read_csv(DATA_DIR / "mc_gbm_paths_universe.csv")

def load_iv_universe():
    return pd.read_csv(DATA_DIR / "implied_vol_universe.csv")

# --- Utilities ---

def get_latest_indicators(indicators, ticker):
    ind = indicators[indicators["ticker"] == ticker].sort_values("date").tail(1)
    return ind

def get_latest_price(ohlcv, ticker):
    last = ohlcv[ohlcv["ticker"] == ticker].sort_values("date").tail(1)
    if last.empty:
        return "not available"
    price = last["close"].values[0]
    return price if pd.notna(price) else "not available"

# --- Technical Indicators ---

def RSI(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss = down.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def MACD(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def create_figure(nrows: int = 2, ncols: int = 2, figsize: Optional[Tuple[float, float]] = (10, 6), sharex: bool = True):
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex)
    return fig, axs

def apply_style_to_figure(fig, grid: bool = True, tick_fontsize: int = 8):
    for ax in fig.get_axes():
        if grid:
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.tick_params(labelsize=tick_fontsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()

# --- 2x2 Technical Indicator Panel ---

def plot_indicators_improved(
    ohlcv: pd.DataFrame,
    tk: str,
    bb_period: int = 20,
    rsi_period: int = 14,
    lookback: Optional[int] = 252,
    recent_vol_days: Optional[int] = 90,
    return_fig: bool = True,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = (10, 6),
):
    df = ohlcv.loc[ohlcv["ticker"] == tk].copy()
    if df.empty:
        raise ValueError(f"No data for ticker {tk}")

    df.index = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_index()
    price_col = "adj_close" if "adj_close" in df.columns else "close"
    if price_col not in df.columns:
        raise ValueError("ohlcv must contain 'adj_close' or 'close'")

    s = df[price_col].dropna()
    if s.empty:
        raise ValueError("No non-null price data for ticker")

    if isinstance(lookback, int) and lookback > 0:
        s = s.tail(lookback)
        df = df.loc[s.index]

    sma = s.rolling(window=bb_period, min_periods=bb_period).mean()
    std = s.rolling(window=bb_period, min_periods=bb_period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std

    rsi = RSI(s, period=rsi_period)
    macd_line, macd_signal, macd_hist = MACD(s)
    vol_series = df["volume"].reindex(s.index) if "volume" in df.columns else pd.Series(dtype=float, index=s.index)

    fig, axs = create_figure(2, 2, figsize=figsize, sharex=True)
    ax00, ax01, ax10, ax11 = axs[0,0], axs[0,1], axs[1,0], axs[1,1]

    # Bollinger Bands
    ax = ax00
    ax.plot(s.index, s, label="Price", linewidth=1)
    ax.plot(sma.index, sma, label=f"SMA{bb_period}", linewidth=1)
    ax.plot(upper.index, upper, linestyle="--", linewidth=1, label="BB Upper")
    ax.plot(lower.index, lower, linestyle="--", linewidth=1, label="BB Lower")
    ax.set_title(f"{tk} Bollinger Bands")
    ax.legend(loc="upper left", fontsize=9)

    # RSI
    ax = ax01
    ax.plot(rsi.index, rsi, color="dimgray", label=f"RSI({rsi_period})", linewidth=1.5)
    ax.axhline(70, color="red", linestyle="--", linewidth=1)
    ax.axhline(30, color="green", linestyle="--", linewidth=1)
    ax.set_title(f"{tk} RSI({rsi_period})")
    ax.legend(loc="upper left", fontsize=9)

    # MACD
    ax = ax10
    ax.plot(macd_line.index, macd_line, label="MACD Line", linewidth=1)
    ax.plot(macd_signal.index, macd_signal, label="MACD Signal", linewidth=1)
    macd_hist_filled = macd_hist.fillna(0)
    hist_vals = macd_hist_filled.values
    colors = np.where(hist_vals >= 0, "green", "red")
    ax.bar(macd_hist_filled.index, hist_vals, width=1, color=colors, alpha=0.6, label="MACD Hist")
    ax.set_title(f"{tk} MACD")
    ax.legend(loc="upper left", fontsize=9)

    # Volume
    ax = ax11
    vol_vals = vol_series.dropna()
    if vol_vals.size > 0:
        try:
            ymax = float(np.nanpercentile(vol_vals.tail(recent_vol_days), 99.99))
        except Exception:
            ymax = float(vol_vals.max())
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = float(vol_vals.max() if vol_vals.size > 0 else 1.0)
    else:
        ymax = 1.0

    ax.bar(vol_series.index, vol_series.values, width=1, color="steelblue", alpha=0.85, label="Volume")
    ax.set_ylim(0, ymax * 1.05)
    ax.set_title(f"{tk} Volume")
    ax.legend(loc="upper left", fontsize=9)

    apply_style_to_figure(fig, grid=True, tick_fontsize=8)
    fig.suptitle(f"Indicators for {tk}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if return_fig:
        return fig, axs
    else:
        plt.show()
        return None

# --- Candlestick Plot ---

def plot_candlestick(
    ohlcv: pd.DataFrame,
    tk: str,
    lookback: int = 60,
    figsize: Optional[Tuple[float, float]] = (12, 4),
    mav: Tuple[int, ...] = (20,),
    style: str = "yahoo",
    apply_style: bool = True,
) -> Tuple[plt.Figure, list]:
    candle_df = (
        ohlcv[ohlcv["ticker"] == tk][["date", "open", "high", "low", "close", "volume"]]
        .copy()
        .set_index("date")
        .sort_index()
        .tail(lookback)
    )
    candle_df.index = pd.to_datetime(candle_df.index)

    if candle_df.empty:
        raise ValueError(f"No OHLCV data for {tk} to plot candlestick.")

    fig, axes = mpf.plot(
        candle_df.rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        ),
        type="candle",
        style=style,
        mav=mav,
        volume=True,
        title=f"{tk} Candlestick Chart (Last {lookback} Days)",
        ylabel="Price",
        ylabel_lower="Volume",
        show_nontrading=False,
        returnfig=True,
        figsize=figsize,
    )

    if apply_style:
        apply_style_to_figure(fig, grid=True, tick_fontsize=8)

    try:
        fig.tight_layout()
    except RuntimeError:
        pass

    # Rotate x labels on bottom axis for readability
    axes_list = axes if isinstance(axes, (list, tuple)) else [axes]
    bottom_ax = axes_list[-1]
    for lbl in bottom_ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")

    # Format volume axis
    vol_ax = axes_list[-1]
    vol_ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
    vol_ax.set_ylabel("Volume (M)", fontsize=10)

    return fig, axes

# --- Portfolio Return Analytics ---

def compute_capm_betas(returns: pd.DataFrame, market_col: str = "SPY") -> pd.DataFrame:
    results = []
    for tk in returns.columns:
        if tk == market_col:
            continue
        y = returns[tk].dropna()
        x = returns[market_col].reindex_like(y).dropna()
        common = y.index.intersection(x.index)
        y = y.loc[common]
        x = x.loc[common]
        if y.empty or x.empty:
            results.append({"Ticker": tk, "Beta": np.nan, "Alpha": np.nan})
            continue
        X = sm.add_constant(x)
        capm = sm.OLS(y, X, missing="drop").fit()
        results.append({
            "Ticker": tk,
            "Beta": float(capm.params.get(market_col, np.nan)),
            "Alpha": float(capm.params.get("const", np.nan)),
        })
    return pd.DataFrame(results).set_index("Ticker")

def plot_portfolio_returns_panel(
    returns: pd.DataFrame,
    capm_df: pd.DataFrame,
    asset: str = "AAPL",
    market_col: str = "SPY",
    lookback: int = 252,
    port_vol_window: int = 60,
    figsize: Tuple[float, float] = (12, 5)
):
    returns_lookback = returns.tail(lookback)
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    ax_scatter, ax_cum = axs

    # Scatter
    if asset in returns_lookback.columns and market_col in returns_lookback.columns:
        y = returns_lookback[asset].dropna()
        x = returns_lookback[market_col].reindex_like(y).dropna()
        common = y.index.intersection(x.index)
        y = y.loc[common]
        x = x.loc[common]
        if not y.empty and not x.empty and asset in capm_df.index:
            ax_scatter.scatter(x, y, alpha=0.5, label=f"{asset} vs {market_col}")
            beta, alpha = capm_df.loc[asset, ["Beta", "Alpha"]]
            x_vals = np.linspace(x.min(), x.max(), 200)
            ax_scatter.plot(x_vals, alpha + beta * x_vals, color="red", label="CAPM fit")
            ax_scatter.set_xlabel(f"{market_col} Return")
            ax_scatter.set_ylabel(f"{asset} Return")
            ax_scatter.set_title(f"{asset} vs {market_col}: Beta/Alpha")
            ax_scatter.legend(fontsize=9)
        else:
            ax_scatter.text(0.5, 0.5, f"Insufficient data for {asset} vs {market_col}",
                            ha="center", va="center", transform=ax_scatter.transAxes)
            ax_scatter.set_title(f"{asset} vs {market_col}: Beta/Alpha")
    else:
        ax_scatter.text(0.5, 0.5, f"{asset} or {market_col} missing", ha="center", va="center", transform=ax_scatter.transAxes)
        ax_scatter.set_title(f"{asset} vs {market_col}: Beta/Alpha")

    # Portfolio Returns
    vol = returns_lookback.rolling(port_vol_window).std().iloc[-1]
    n_assets = returns_lookback.shape[1]
    w_eq = pd.Series(1.0 / n_assets, index=returns_lookback.columns)
    inv_vol = 1.0 / vol.replace(0, np.nan)
    if not np.isfinite(inv_vol).any() or inv_vol.isna().all():
        w_vw = w_eq.copy()
    else:
        inv_vol = inv_vol.fillna(inv_vol.mean())
        w_vw = inv_vol / inv_vol.sum()
        w_vw = w_vw.reindex(returns_lookback.columns).fillna(0)

    port_eq = (returns_lookback * w_eq).sum(axis=1)
    port_vw = (returns_lookback * w_vw).sum(axis=1)
    cum = pd.DataFrame({"Equal": port_eq, "Vol-Weighted": port_vw}).add(1).cumprod()

    if asset in returns_lookback.columns:
        cum_asset = returns_lookback[asset].add(1).cumprod()
        ax_cum.plot(cum_asset.index, cum_asset, label=f"{asset} (single)", color="tab:orange", linestyle="--", linewidth=2)

    ax_cum.plot(cum.index, cum["Equal"], label="Equal Portfolio", color="tab:blue", linewidth=1.5)
    ax_cum.plot(cum.index, cum["Vol-Weighted"], label="Vol-Weighted Portfolio", color="tab:green", linewidth=1.5)
    ax_cum.set_title("Cumulative Returns: Portfolios vs Asset")
    ax_cum.legend(fontsize=9)
    ax_cum.set_xlabel("Date")
    ax_cum.set_ylabel("Cumulative Return")

    apply_style_to_figure(fig, grid=True, tick_fontsize=8)
    fig.tight_layout()
    return fig, axs

# --- Option Analytics (For Options Panel) ---

# --- DATA LOADERS ---

def load_delta_hedge():
    return pd.read_csv(DATA_DIR / "delta_hedge_full_universe_30d.csv", parse_dates=["date"])

def load_binomial_prices():
    return pd.read_csv(DATA_DIR / "binomial_prices_full_universe.csv")

def load_mc_option_prices():
    return pd.read_csv(DATA_DIR / "mc_option_prices_full_universe.csv")

def load_mc_gbm_paths():
    return pd.read_csv(DATA_DIR / "mc_gbm_paths_universe.csv")

def load_iv_universe():
    # One merged table (call & put) with opt_type column
    return pd.read_csv(DATA_DIR / "implied_vol_universe.csv")


# --- GREEKS TABLE ---

def get_latest_greeks(delta_hedge_df, ticker):
    df = delta_hedge_df[delta_hedge_df["ticker"] == ticker]
    if df.empty:
        return pd.DataFrame()
    latest = df.sort_values("date").iloc[-1]
    fields = [f for f in ["delta", "gamma", "vega", "theta", "rho"] if f in df.columns]
    greeks = {f.capitalize(): latest.get(f, np.nan) for f in fields}
    return pd.DataFrame([greeks])

# --- DELTA-HEDGE P&L PLOT ---

def plot_delta_hedge_pnl(delta_hedge_df, ticker):
    df = delta_hedge_df[delta_hedge_df["ticker"] == ticker]
    if df.empty or "hedge_pnl" not in df:
        return None
    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    ax.plot(df["date"], df["hedge_pnl"], label="Delta-hedge P&L")
    ax.axhline(0, lw=1, color="k", alpha=0.5)
    strike = df["strike"].iloc[0] if "strike" in df.columns else np.nan
    # SHORT, CONSISTENT TITLE
    ax.set_title(f"Delta-Hedge P&L ({ticker}, K={strike:.2f})", fontsize=11)
    ax.set_xlabel("Date")
    ax.set_ylabel("P&L (model-marked)")
    ax.legend()
    fig.tight_layout()
    return fig

# --- BINOMIAL PRICES VS STRIKE PLOT ---


def plot_binomial_vs_strike(binomial_df, ticker):
    df = binomial_df[binomial_df["ticker"] == ticker]
    if df.empty:
        return None
    S = df["K"].mean()
    sigma = df["sigma"].iloc[0]
    fig, ax = plt.subplots(figsize=(7,4), dpi=130)
    ax.plot(df["K"], df["Euro_Call"], label="Euro Call")
    ax.plot(df["K"], df["Amer_Call"], label="Amer Call", linestyle="--")
    ax.plot(df["K"], df["Euro_Put"],  label="Euro Put")
    ax.plot(df["K"], df["Amer_Put"],  label="Amer Put", linestyle="--")
    # SHORT, CONSISTENT TITLE (no linebreak!)
    ax.set_title(f"Binomial Prices ({ticker})", fontsize=10)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Option Price")
    ax.legend()
    fig.tight_layout()
    return fig

# --- MONTE CARLO GBM SPAGHETTI & HISTOGRAM PANEL ---

def plot_mc_gbm_panel(mc_gbm_df, ticker):
    gbm = mc_gbm_df[mc_gbm_df["ticker"] == ticker]
    if gbm.empty:
        return None, None
    time_cols = [c for c in gbm.columns if c.startswith('t')]
    t_grid = np.arange(len(time_cols))
    paths = gbm[time_cols].apply(pd.to_numeric, errors="coerce").ffill(axis=1).values[:30]  # show up to 30 paths

    fig1, ax1 = plt.subplots(figsize=(7, 4), dpi=130)
    for i in range(paths.shape[0]):
        ax1.plot(t_grid, paths[i], alpha=0.6)
    ax1.set_title(f"Monte Carlo GBM Simulated Paths ({ticker})", fontsize=12)
    ax1.set_xlabel("Time (steps)")
    ax1.set_ylabel("Price")
    fig1.tight_layout()

    # Terminal distribution
    terminal_vals = paths[:, -1]
    fig2, ax2 = plt.subplots(figsize=(7, 4), dpi=130)
    ax2.hist(terminal_vals, bins=30, density=True, alpha=0.7)
    ax2.set_title(f"GBM Terminal Price Distribution ({ticker})", fontsize=12)
    ax2.set_xlabel("Terminal Price")
    ax2.set_ylabel("Density")
    fig2.tight_layout()
    return fig1, fig2

# --- MONTE CARLO OPTION SUMMARY TABLE ---

def get_mc_option_summary(mc_option_df, ticker):
    df = mc_option_df[mc_option_df["ticker"] == ticker]
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(index=["Instrument"], values=["MC_Price", "MC_StdErr", "BS_ClosedForm"]).round(4).reset_index()

# --- IMPLIED VOLATILITY TABLE ---

def get_iv_table(iv_df, ticker):
    df = iv_df[iv_df["ticker"] == ticker]
    if df.empty:
        return pd.DataFrame()
    # Merge call/put, show expiry, strike, opt_type, IV
    show = df[["opt_type", "expiry", "K", "price_input", "IV_Newton"]].copy()
    show = show.rename(columns={"opt_type":"Type", "expiry":"Expiry", "K":"Strike", "price_input":"Option Price", "IV_Newton":"Implied Volatility"})
    show = show.sort_values(["Expiry", "Type", "Strike"])
    return show.round(4)
