#!/usr/bin/env python3
import os, json, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from polygon import RESTClient
from polygon.rest.models import TickerNews

API_KEY = os.getenv("POLYGON_API_KEY")

# Start from April 23, 2025
START = "2025-07-22T11:25:00Z"
END = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# All tickers we want to track
TICKERS = [
    'AAPL', 'MSFT', 'TSLA', 'GOOG', 'META', 'NVDA',
    'RBLX', 'AMC', 'GME', 'CROX',
    'ANF', 'URBN', 'AEO', 'LEVI', 'GPS', 'ASO', 'LULU', 'NKE',
    'JNJ', 'PFE', 'UNH', 'MRK', 'LLY', 'ABT'
]

# Output directory
OUT_DIR = Path("/Users/saahi/social-arbitrage-trader/data/news")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEEP_FIELDS = (
    "id","title","article_url","published_utc","updated_utc",
    "author","description","tickers","publisher","insights",
    "image_url","amp_url","keywords"
)

# Rate limiting and window settings
SLEEP_SEC = 15  # Time between API calls
WINDOW_DAYS = 14  # Size of each time window

def to_plain(o):
    """Recursively turn Polygon models / objects into JSON-serializable types."""
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, list):
        return [to_plain(x) for x in o]
    if isinstance(o, dict):
        return {k: to_plain(v) for k, v in o.items()}
    if hasattr(o, "model_dump"):
        return to_plain(o.model_dump())
    if hasattr(o, "__dict__"):
        return to_plain(vars(o))
    return str(o)

def flatten(news_obj: TickerNews) -> dict:
    d = to_plain(news_obj)
    return {k: d.get(k) for k in KEEP_FIELDS if k in d}

def is_primary_ticker(item: dict, ticker: str) -> bool:
    tix = item.get("tickers") or []
    return len(tix) > 0 and tix[0] == ticker

def get_time_windows(start_str: str, end_str: str):
    """Break time range into 2-week windows."""
    start = datetime.strptime(start_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    
    windows = []
    current = start
    
    while current < end:
        window_end = min(current + timedelta(days=WINDOW_DAYS), end)
        windows.append((
            current.strftime("%Y-%m-%dT%H:%M:%SZ"),
            window_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        ))
        current = window_end
    
    return windows

def fetch_news_for_ticker(client: RESTClient, ticker: str, start_time: str, end_time: str) -> list:
    """Fetch news for a specific ticker in a time window."""
    print(f"\nFetching {ticker} news from {start_time} to {end_time}")
    rows = []
    
    try:
        for n in client.list_ticker_news(
            ticker=ticker,
            published_utc_gte=start_time,
            published_utc_lte=end_time,
            sort="published_utc",
            order="asc",
            limit=1000
        ):
            if isinstance(n, TickerNews):
                item = flatten(n)
                if is_primary_ticker(item, ticker):
                    rows.append(item)
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return []
        
    print(f"Found {len(rows)} articles for {ticker}")
    return rows

def save_ticker_data(ticker: str, articles: list):
    """Save articles for a ticker to a JSON file."""
    if not articles:
        return
        
    out_path = OUT_DIR / f"{ticker}.json"
    
    # Load existing data if any
    existing = []
    if out_path.exists():
        with out_path.open('r') as f:
            existing = json.load(f)
    
    # Merge new articles with existing ones, avoiding duplicates
    seen_ids = {a['id'] for a in existing}
    merged = existing + [a for a in articles if a['id'] not in seen_ids]
    
    # Sort by published date
    merged.sort(key=lambda x: x.get('published_utc', ''))
    
    with out_path.open('w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"Saved {len(merged)} total articles for {ticker}")

def main():
    if not API_KEY:
        raise SystemExit("POLYGON_API_KEY not set")

    client = RESTClient(API_KEY)
    windows = get_time_windows(START, END)
    
    print(f"Processing {len(TICKERS)} tickers across {len(windows)} time windows")
    
    for ticker in TICKERS:
        ticker_articles = []
        
        for start, end in windows:
            articles = fetch_news_for_ticker(client, ticker, start, end)
            ticker_articles.extend(articles)
            
            # Rate limiting sleep between API calls
            time.sleep(SLEEP_SEC)
        
        save_ticker_data(ticker, ticker_articles)

if __name__ == "__main__":
    main()
