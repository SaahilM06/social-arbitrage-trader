import requests
import pandas as pd
import json
import re


def categorize_message(msg):
    text = msg.lower()
    categories = {
        "bullish_examples": [],
        "bearish_examples": [],
        "technical_signals": [],
        "price_targets": [],
        "macro_commentary": [],
        "promotions": []
    }

    # Naive example keyword rules:
    if any(word in text for word in ["to the moon", "rocket", "🚀", "buy", "bullish"]):
        categories["bullish_examples"].append(msg)
    if any(word in text for word in ["short", "bearish", "sell", "crash"]):
        categories["bearish_examples"].append(msg)
    if any(word in text for word in ["support", "resistance", "moving average", "breakout", "ema"]):
        categories["technical_signals"].append(msg)
    if re.search(r"\b\d{2,4}\b", text):
        categories["price_targets"].append(msg)
    if any(word in text for word in ["macro", "inflation", "interest rates", "fed", "economy"]):
        categories["macro_commentary"].append(msg)
    if any(word in text for word in ["promotion", "club", "discord", "follow"]):
        categories["promotions"].append(msg)

    return categories


def get_stocktwits_messages(symbol, limit=30):
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
    }
    response = requests.get(url, headers=headers)
    data = response.json()

    # Collect categorized messages
    organized = {
        symbol: {
            "sentiment": "mixed",
            "bullish_examples": [],
            "bearish_examples": [],
            "technical_signals": [],
            "price_targets": [],
            "macro_commentary": [],
            "promotions": [],
            "related_tickers": [],  # Optional: Parse later
            "volume_change": None,
            "sentiment_change": None
        }
    }

    for msg in data["messages"][:limit]:
        body = msg["body"]
        categorized = categorize_message(body)

        for key in categorized:
            organized[symbol][key].extend(categorized[key])

    # Remove duplicates in case messages overlap categories
    for key in organized[symbol]:
        if isinstance(organized[symbol][key], list):
            organized[symbol][key] = list(set(organized[symbol][key]))

    with open("/Users/saahi/social-arbitrage-trader/data/processed/sentiment_data.json", "w") as f:
        json.dump(organized, f, indent=4)

    print(f"Saved {symbol} sentiment data to sentiment_data.json")


get_stocktwits_messages('AAPL')
