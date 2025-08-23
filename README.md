# Finance Metrics Pipeline

A modular pipeline to ingest, store, analyze, and visualize financial market data using Python, Postgres, Jupyter, and Streamlit. Designed for iterative, collaborative research and rapid prototyping.

---

## Team

- Shervin Taheripour
- Dariya Sharonova

---

## Project Phases (short)

1. Ingest financial data (yfinance, Alpha Vantage)
2. Persist masterframes to local storage / Postgres (Docker Compose) or local .csv storage
3. Analyze in Jupyter notebooks (develop plots & functions)
4. Prototype Streamlit dashboard (single app that reuses notebook functions)
5. **(Optional)** Expose REST API (Flask)
6. **(Optional)** Containerize, Automate & Deploy (CI/CD, Airflow, Kubernetes)

---

## Folder Overview

```
finance-pipeline/
├── ingestion/
│   ├── alpha_vantage/alpha_v_methods.py    # Alpha Vantage ingestion utilities
│   ├── yfinance/yfinance_methods_v2.py     # yfinance ingestion utilities (v2)
│   └── .env                                # API/database keys (local)
├── analysis/
│   ├── data/                               # local data cache used by notebooks (analysis/data)
│   ├── masterframes_ingestion.ipynb        # Build & persist masterframes (ohlcv, dividends, splits, meta)
│   ├── eda.ipynb                           # Universe metadata, single-ticker snapshot, teaser plots
│   ├── statistical_financial_analysis.ipynb# Indicators, heatmap, rolling correlations, CAPM, portfolio sims
│   ├── option_pricing_models.ipynb         # IV smile / term-structure for a chosen ticker + pricing helper
│   ├── ml_models.ipynb                     # Optional minimal models (lag features, baseline regressions)
│   ├── indicators_metrics.ipynb            # Compute and export indicators (SMA/EMA/RSI/MACD/vol/rollcorr)
│   └── dashboard_prototyping.ipynb         # Small library of plotting functions to port to Streamlit
├── dashboard/                              # Streamlit app (single app that uses analysis/data)
│   ├── app.py                              # (to be created) Main Streamlit app
│   └── utils.py                            # (to be created) KPI helpers and plotting wrappers
├── data/                                   # (optional) alternative project-level data folder if used
├── api/                                    # Optional Flask API
├── db/docker-compose.yaml                  # Postgres via Docker Compose
├── requirements.txt                        # Python dependencies
└── README.md (this file)                   # Project overview and plan
```

**Note:** Current notebook workflow expects and uses `analysis/data/` as the local cache directory for CSV/Parquet files (e.g., `analysis/data/ohlcv.csv`, `analysis/data/indicators.csv`). If you prefer a repo-root `data/` instead, you can switch by updating the small path block at the top of each notebook.

---

## Workflow: End-to-End Artefact Lineage

**Run these notebooks/scripts IN ORDER for a fresh pipeline:**

1. ### Data Ingestion
    - `analysis/masterframes_ingestion.ipynb`  
      *Downloads and standardizes raw data.*
    - **Artefacts:**  
      - `ohlcv.csv`  
      - `dividends.csv`  
      - `splits.csv`  
      - `earnings_dates.csv`  
      - `meta_kv.csv`

2. ### Indicators & Metrics
    - `analysis/indicators_metrics.ipynb`  
      *Calculates rolling statistics and trading indicators.*
    - **Artefacts:**  
      - `indicators.csv`

3. ### Statistical & Portfolio Analysis
    - `analysis/statistical_financial_analysis.ipynb`  
      *(Prepares, tests, and visualizes all time series logic used in dashboard, but does not save artefacts.)*

4. ### Options & Pricing Models
    - `analysis/option_pricing_models.ipynb`  
      *Calculates greeks, binomial & MC pricing, and implied volatility across universe.*
    - **Artefacts:**  
      - `delta_hedge_full_universe_30d.csv`  
      - `binomial_prices_full_universe.csv`  
      - `mc_option_prices_full_universe.csv`  
      - `mc_gbm_paths_universe.csv`  
      - `implied_vol_universe.csv`

5. ### Dashboard Prototyping (Optional)
    - `analysis/dashboard_prototyping.ipynb`  
      *Test run: verifies all artefacts and plot functions work with real data, before going to Streamlit.*

6. ### Streamlit Dashboard
    - `dashboard/app.py`, `dashboard/utils.py`  
      *Loads all artefacts from `analysis/data/` and renders visuals and tables interactively.*

---

## To Reproduce the Dashboard

1. **Run `masterframes_ingestion.ipynb` to create all base CSVs.**
2. **Run `indicators_metrics.ipynb` to get rolling technicals.**
3. **Run `option_pricing_models.ipynb` to generate all options and simulation artefacts.**
4. *(Recommended)*: Open `dashboard_prototyping.ipynb` and visually verify plots for a ticker like `AAPL`.
5. **Launch the dashboard:**
    ```bash
    cd dashboard
    streamlit run app.py
    ```
6. **Re-run the pipeline as new data arrives (e.g. schedule these in order for automation).**

---

## Artefact Manifest

All outputs are saved in `analysis/data/` and automatically loaded by the dashboard:

| Artefact                        	| Created By                               | Used For (Dashboard Block)       |
|-----------------------------------|------------------------------------------|----------------------------------|
| ohlcv.csv                       	| masterframes_ingestion                   | All charts/tables                |
| meta_kv.csv                     	| masterframes_ingestion                   | Sidebar, metadata panel          |
| indicators.csv                   	| indicators_metrics                       | Technical indicators panel       |
| delta_hedge_full_universe_30d.csv	| option_pricing_models                    | Option: Greeks, P&L plot         |
| binomial_prices_full_universe.csv	| option_pricing_models                    | Option: Binomial plot            |
| mc_option_prices_full_universe.csv| option_pricing_models                    | Option: MC summary table         |
| mc_gbm_paths_universe.csv        	| option_pricing_models                    | Option: MC GBM plot, histogram   |
| implied_vol_universe.csv         	| option_pricing_models                    | Option: IV summary table         |

---

## Notes

- Always check the artefacts in `analysis/data/` after each notebook.  
- If the dashboard errors, usually it’s missing or misnamed artefact columns.
- For automation (Docker, Airflow, etc), see `pipeline_pseudocode.md` for wiring.

---

## Credits & License

- Built by Dariya Sharonova, Shervin Taheripour (2025).
- Not affiliated with any financial institution.  
- MIT License.
