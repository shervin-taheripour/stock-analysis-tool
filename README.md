# Finance Metrics Pipeline

A modular pipeline to ingest, store, analyze, and visualize financial market data using Python, Postgres, Jupyter, and Streamlit. Designed for iterative, collaborative research and development.

---

## Team

- Shervin Taheripour
- Dariya Sharonova

---

## Project Phases

1. **Ingest Financial Data**

   - APIs: yfinance, Alpha Vantage *(Finnhub on hold: premium required for history)*
   - Modular ingestion scripts per source

2. **Store in Postgres**

   - Local Postgres (later via Docker Compose)

3. **Analyze in Jupyter**

   - Notebooks using pandas, numpy, matplotlib, seaborn

4. **Visualize with Streamlit**

   - Interactive dashboard for financial metrics

5. **Containerize**

   - Docker for DB, later for app components

6. **Store Results**

   - Metrics/results persisted in Postgres

7. **(Optional)** Expose REST API (Flask)

8. **(Optional)** Automate & Deploy (CI/CD, Airflow, Kubernetes)

---

## Folder Overview

```
finance-pipeline/
├── ingestion/
│   ├── alpha_vantage/alpha_v_methods.py    # Alpha Vantage ingestion utilities
│   ├── yfinance/yfinance_methods.py        # yfinance ingestion utilities
│   └── .env                                # All API/database keys
├── analysis/analysis.ipynb                 # Main analysis notebook
├── dashboard/                              # Streamlit dashboard app
├── api/                                    # Optional Flask API
├── db/docker-compose.yaml                  # Postgres via Docker Compose
├── k8s/                                    # Kubernetes manifests (optional)
├── ci-cd/                                  # CI/CD configs
└── README.md
```

---

## Requirements

- Python 3.10+
- All dependencies listed in `requirements.txt`

---

## Getting Started

1. **Clone and Navigate**

   ```bash
   git clone https://github.com/shervin-taheripour/stock-analysis-tool.git
   cd stock-analysis-tool
   ```

2. **Set Up Python Environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Start Postgres via Docker**

   ```bash
   cd db
   docker-compose up -d
   ```

4. **Create **``** (in ingestion/)**

   ```env
   # ingestion/.env
   DB_URL=postgresql://postgres:mypw1234@localhost:5432/finance_db
   FINNHUB_KEY=your_finnhub_key
   ALPHAVANTAGE_KEY=your_alpha_vantage_key
   ```

5. **Run Ingestion (from notebooks/scripts)**

   - Import and call ingestion methods in your Jupyter notebooks:
     ```python
     import sys
     sys.path.append('../ingestion/alpha_vantage')
     from alpha_v_methods import av_get_daily_history
     ```

6. **Start Jupyter for Analysis**

   ```bash
   cd analysis
   jupyter notebook
   ```

7. **Launch Streamlit Dashboard**

   ```bash
   cd dashboard
   streamlit run app.py
   ```

---

## Notes

- `.env` is required but **should not be committed to git**.
- All API/database keys are managed in a single `.env` for simplicity.
- Docker ensures consistent environments; code locally, then containerize.
- CI/CD and orchestration are planned for later phases.

---

## License

MIT

