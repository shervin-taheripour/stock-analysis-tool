CREATE TABLE analyst_recommendations (
    ticker TEXT,
    period TEXT,
    strong_buy INTEGER,
    buy INTEGER,
    hold INTEGER,
    sell INTEGER,
    strong_sell INTEGER,
    PRIMARY KEY (ticker, period)
);

CREATE TABLE company_metadata (
    ticker TEXT PRIMARY KEY,
    key TEXT,
    value TEXT
);

CREATE TABLE dividends (
    ticker TEXT,
    date DATE,
    dividends NUMERIC,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE earnings_date (
    ticker TEXT,
    earnings_date DATE,
    eps_estimate NUMERIC,
    reported_eps NUMERIC,
    surprise NUMERIC,
    PRIMARY KEY (ticker, earnings_date)
);

CREATE TABLE events_calendar (
    ticker TEXT,
    event_type TEXT,
    event_date TIMESTAMP,
    PRIMARY KEY (ticker, event_type)
);

CREATE TABLE financials_balance (
    ticker TEXT,
    report_date DATE,
    metric_name TEXT,
    value NUMERIC,
    PRIMARY KEY (ticker, report_date, metric_name)
);

CREATE TABLE financials_cashflow (
    ticker TEXT,
    report_date DATE,
    metric_name TEXT,
    value NUMERIC,
    PRIMARY KEY (ticker, report_date, metric_name)
);

CREATE TABLE financials_income (
    ticker TEXT,
    report_date DATE,
    metric_name TEXT,
    value NUMERIC,
    PRIMARY KEY (ticker, report_date, metric_name)
);

CREATE TABLE ohlcv_data_raw (
    ticker TEXT,
    date DATE,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    dividends NUMERIC,
    splits NUMERIC,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE options_chain (
    contract_symbol TEXT PRIMARY KEY,
    last_trade_date TIMESTAMP,
    strike NUMERIC,
    last_price NUMERIC,
    bid NUMERIC,
    ask NUMERIC,
    change NUMERIC,
    percent_change NUMERIC,
    volume INTEGER,
    open_interest INTEGER,
    implied_volatility NUMERIC,
    in_the_money BOOLEAN,
    contract_size TEXT,
    currency TEXT,
    option_type TEXT,
    expiration_date DATE
);

CREATE TABLE splits (
    ticker TEXT,
    date DATE,
    splits NUMERIC,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE sustainability_data (
    ticker TEXT,
    metric_name TEXT,
    esg_score NUMERIC,
    PRIMARY KEY (ticker, metric_name)
);
