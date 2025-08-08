import os
import requests
from bs4 import BeautifulSoup

def save_finviz_html(ticker, output_dir="html_outputs"):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Failed to fetch {ticker}: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save raw HTML
    output_path = os.path.join(output_dir, f"{ticker}_finviz.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(soup.prettify())

    print(f"✅ Saved {ticker} HTML to: {output_path}")

if __name__ == "__main__":
    tickers = ["AAPL"]  # Add more tickers if needed
    for ticker in tickers:
        save_finviz_html(ticker)
