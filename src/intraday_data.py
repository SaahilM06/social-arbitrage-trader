import yfinance as yf
import pandas as pd
import json
import os

INTERVAL = '5m'
TICKERS = [
    'AAPL', 'MSFT', 'TSLA', 'GOOG', 'META', 'NVDA',
    'RBLX', 'AMC', 'GME', 'CROX',

    'ANF', 'URBN', 'AEO', 'LEVI', 'GPS', 'ASO', 'LULU', 'NKE',

    'JNJ', 'PFE', 'UNH', 'MRK', 'LLY', 'ABT'
]
def fetch_and_update_data(symbol):
    df = yf.download(tickers=symbol, period='60d', interval=INTERVAL, progress=False, auto_adjust=False)
    df['percent_change_close'] = df['Close'].pct_change() * 100
    df['percent_change_close'] = df['percent_change_close'].fillna(0)
    output_directory = '/Users/saahi/social-arbitrage-trader/data/processed'
    file_name = f'{output_directory}/{symbol}_ohlcv.json'
    if os.path.exists(file_name) and os.stat(file_name).st_size != 0:
        with open(file_name, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    for index, row in df.iterrows():
        ts = pd.to_datetime(index).strftime("%Y-%m-%d %H:%M:%S")
        existing_data[ts] = {
            'open': round(float(row['Open']), 2),
            'high': round(float(row['High']), 2),
            'low': round(float(row['Low']), 2),
            'close': round(float(row['Close']), 2),
            'volume': int(row['Volume']),
            'percent_change_close': round(float(row['percent_change_close']), 2)
        }
    existing_data = dict(sorted(existing_data.items()))
    
    with open(file_name, 'w') as f:
        json.dump(existing_data, f, indent=4)

    print(f"{symbol} updated.")


for ticker in TICKERS:
    fetch_and_update_data(ticker)
