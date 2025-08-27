

# from yfinance_methods import get_ohlcv_data
import sqlalchemy as db
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Numeric, TIMESTAMP, Date, DATE,  Boolean
import pandas as pd
import yfinance as yf

import sys
sys.path.insert(0, 'ingestion/yfinance')


# INGEST_DIR = Path("../ingestion/yfinance")

class Base(DeclarativeBase):
    pass


# Database configurations
SQLALCHEMY_DATABASE_URL = "sqlite:///./stock_analysis.db"


# SQLAlchemy models

class Ticker(Base):
    __tablename__ = "ticker"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    type: Mapped[str]
    description: Mapped[str]


class Company_metadata(Base):
    __tablename__ = "company_metadata"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker_id = mapped_column(ForeignKey("ticker.id"))
    key: Mapped[str]
    value: Mapped[str]


class Ohlcv_data_raw(Base):
    __tablename__ = "ohlcv_data_raw"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker_id = mapped_column(ForeignKey("ticker.id"), nullable=False)
    date = mapped_column(DateTime)
    open: Mapped[float]
    high: Mapped[float]
    low: Mapped[float]
    close: Mapped[float]
    volume: Mapped[float]
    dividends: Mapped[float]
    splits: Mapped[float]


class Splits(Base):
    __tablename__ = "splits"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker_id = mapped_column(ForeignKey("ticker.id"))
    date = mapped_column(DateTime)
    splits: Mapped[float]


class Options_chain(Base):
    __tablename__ = "options_chain"
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker_id = mapped_column(ForeignKey("ticker.id"))

    contract_symbol: Mapped[str]
    last_trade_date = mapped_column(DateTime)
    strike: Mapped[float]
    last_price: Mapped[float]
    bid: Mapped[float]
    ask: Mapped[float]
    change: Mapped[float]
    percent_change: Mapped[float]
    volume: Mapped[int]
    open_interest: Mapped[int]
    implied_volatility: Mapped[float]
    in_the_money: Mapped[bool]
    contract_size: Mapped[str]
    currency: Mapped[str]
    option_type: Mapped[str]
    expiration_date = mapped_column(DateTime)

class Financials_income(Base):
    __tablename__ = "financials_income"
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker_id = mapped_column(ForeignKey("ticker.id"))
    metric_name: Mapped[str]
    report_date = mapped_column(DateTime)
    value: Mapped[float]

class Financials_balance(Base):
    __tablename__ = "financials_balance"
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker_id = mapped_column(ForeignKey("ticker.id"))
    metric_name: Mapped[str]
    report_date = mapped_column(DateTime)
    value: Mapped[float]

class Financials_cashflow(Base):
    __tablename__ = "financials_cashflow"
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker_id = mapped_column(ForeignKey("ticker.id"))
    metric_name: Mapped[str]
    report_date = mapped_column(DateTime)
    value: Mapped[float]

class Events_calendar(Base):   
    __tablename__ = "events_calendar"
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker_id = mapped_column(ForeignKey("ticker.id"))
    event_type: Mapped[str]
    event_date = mapped_column(DateTime)

class Sustainability_data(Base):
    __tablename__ = "sustainability_data"
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker_id = mapped_column(ForeignKey("ticker.id"))
    metric_name: Mapped[str]
    esg_score: Mapped[float]

class Dividends(Base):
    __tablename__ = "dividends"
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker_id = mapped_column(ForeignKey("ticker.id"))
    date = mapped_column(DateTime)
    dividends: Mapped[float]    

class Earnings_date(Base):
    __tablename__ = "earnings_date"
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker_id = mapped_column(ForeignKey("ticker.id"))
    earnings_date = mapped_column(DateTime)
    eps_estimate: Mapped[float]
    reported_eps: Mapped[float]
    surprise: Mapped[float]

class Analyst_recommendations(Base):
    __tablename__ = "analyst_recommendations"
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker_id = mapped_column(ForeignKey("ticker.id"))
    period: Mapped[str]
    strong_buy: Mapped[int]
    buy: Mapped[int]
    hold: Mapped[int]
    sell: Mapped[int]
    strong_sell: Mapped[int]    

   

engine = create_engine(SQLALCHEMY_DATABASE_URL)

# conn = engine.connect()

Base.metadata.create_all(engine)


with Session(engine) as session:
    session.add(Ticker(name="AAPL", type="Tech Stock",
                description="Apple - large-cap tech"))
    session.add(Ticker(name="MSFT", type="Tech Stock",
                description="Microsoft - another major tech"))
    session.add(Ticker(name="NVDA", type="Tech Stock",
                description="NVIDIA - semiconductors, high beta"))
    session.add(Ticker(name="AMZN", type="Consumer",
                description="Amazon - e-commerce & cloud"))
    session.add(Ticker(name="TSLA", type="Consumer",
                description="Tesla - high volatility, growth"))
    session.add(Ticker(name="META", type="Tech Stock",
                description="Meta - social media and ad-driven"))
    session.add(Ticker(name="GOOG", type="Tech Stock",
                description="Alphabet - search, ads, cloud"))
    session.add(Ticker(name="JPM", type="Financial",
                description="JPMorgan - big bank exposure"))
    session.add(Ticker(name="V", type="Financial",
                description="Visa - payments and macro cycles"))
    session.add(Ticker(name="XOM", type="Energy",
                description="ExxonMobil - classic energy play"))
    session.add(Ticker(name="CVX", type="Energy",
                description="Chevron - more stable than XOM"))
    session.add(Ticker(name="UNH", type="Healthcare",
                description="UnitedHealth - defensive sector"))
    session.add(Ticker(name="JNJ", type="Healthcare",
                description="Johnson & Johnson - pharma/stable"))
    session.add(Ticker(name="PFE", type="Healthcare",
                description="Pfizer - biotech + COVID cycle"))
    session.add(Ticker(name="NKE", type="Consumer",
                description="Nike - global consumer brand"))
    session.add(Ticker(name="XLF", type="ETF",
                description="Financial sector ETF"))
    session.add(Ticker(name="XLK", type="ETF", description="Tech sector ETF"))
    session.add(Ticker(name="SPY", type="ETF",
                description="S&P 500 - benchmark index"))
    session.add(Ticker(name="QQQ", type="ETF",
                description="Nasdaq 100 - tech-heavy benchmark"))
    session.add(Ticker(name="VIXY", type="ETF",
                description="Short-term volatility index (proxy for VIX)"))
    session.commit()

print("Tables created and tickers added.")
