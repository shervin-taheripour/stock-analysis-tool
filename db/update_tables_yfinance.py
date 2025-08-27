# filepath: c:\Users\darin\greenbootcamps\stock_analysis_tool\db\update_tables_yfinance.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ingestion.yfinance.yfinance_methods import get_ohlcv_data_db, get_options_chain_db, get_company_metadata_db
from create_tables_yfinance import Ticker, Splits  # <-- Add this line
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import Session
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import create_engine
import sqlalchemy as db


SQLALCHEMY_DATABASE_URL = "sqlite:///./stock_analysis.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)

factory = sessionmaker(bind=engine)
session = factory()

# def call_update_tables():
with Session(engine) as session:
    tickers = session.query(Ticker).all()
    if len(tickers) > 0:
        for t in tickers:
            df = get_options_chain_db(t.name, t.id, None)
            df.to_sql('options_chain', engine, index=False, if_exists="replace")
            session.commit()
            print(f"✅ Updated options_chain table for ticker: {t.name}")

            # df = get_company_metadata_db(t.name, t.id)
            # df.to_sql('company_metadata', engine, index=False, if_exists="replace")
            # session.commit()
            # print(f"✅ Updated company_metadata table for ticker: {t.name}")

            # df = get_ohlcv_data_db(t.name, t.id, period='5y', interval='1d')
            # df.to_sql('company_metadata', engine, index=False, if_exists="replace")
            # session.commit()
            # print(f"✅ Updated company_metadata table for ticker: {t.name}")
