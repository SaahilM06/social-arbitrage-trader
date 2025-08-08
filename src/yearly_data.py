import requests
import pandas as pd
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
if not API_KEY:
    raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set. Please add it to your .env file.")

TICKER = 'PFE'

def get_ohlcv_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()

    time_series = data.get('Time Series (Daily)', {})

    if not time_series:
        print("No data returned.")
        return
    
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df = df.rename(columns={
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. volume': 'volume'
    })

    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    df.sort_index(inplace=True)

    # Calculate % change on closing prices
    df['percent_change_close'] = df['close'].pct_change() * 100
    df['percent_change_close'].fillna(0, inplace=True)

    ohlcv_dict = {
        idx.strftime("%Y-%m-%d %H:%M:%S"): {
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': int(row['volume']),
            'percent_change_close': round(row['percent_change_close'], 2)
        }
        for idx, row in df.iterrows()
    }

    output_directory = '/Users/saahi/social-arbitrage-trader/data/processed'
    with open(f'{output_directory}/{symbol}_ohlcv.json', 'w') as f:
        json.dump(ohlcv_dict, f, indent=4)

    print(f"Saved OHLCV data with % change for {symbol} to {symbol}_ohlcv.json")

get_ohlcv_data(TICKER)
