import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json
import os
import time 
import random
import logging
from transformers import pipeline


# Configure logging
logging.basicConfig(
    filename='finviz_scraper.log',
    level=logging.DEBUG,  # Changed from INFO to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # This will clear the log file each time we run the script
)

# Initialize FinBERT sentiment analyzer
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    logging.info("Successfully loaded FinBERT model")
except Exception as e:
    logging.error(f"Error loading FinBERT model: {str(e)}")
    raise

TICKERS = [
    'AAPL', 'MSFT', 'TSLA', 'GOOG', 'META', 'NVDA',
    'RBLX', 'AMC', 'GME', 'CROX',

    'ANF', 'URBN', 'AEO', 'LEVI', 'GPS', 'ASO', 'LULU', 'NKE',

    'JNJ', 'PFE', 'UNH', 'MRK', 'LLY', 'ABT'
]

OUTPUT_DIR = '/Users/saahi/social-arbitrage-trader/data/sentiment/finviz_json'

def analyze_sentiment(text):
    """
    Analyze sentiment of a text using FinBERT.
    Returns a dictionary with label and score.
    """
    try:
        result = sentiment_analyzer(text)[0]
        return {
            "label": result["label"],
            "score": float(result["score"])  # Convert numpy float to Python float for JSON serialization
        }
    except Exception as e:
        logging.error(f"Error analyzing sentiment for text: {text[:100]}... Error: {str(e)}")
        return {
            "label": "unknown",
            "score": 0.0
        }

def parse_finviz_time(raw_time):
    # Set your future start date
    FUTURE_START = datetime(2025, 4, 23)
    
    try:
        raw_time = raw_time.strip()
        
        if "Today" in raw_time:
            # Handle "Today HH:MMam/pm"
            time_part = raw_time.replace("Today", "").strip()
            dt = datetime.strptime(time_part, "%I:%M%p")
            base_date = FUTURE_START if FUTURE_START > datetime.now() else datetime.now()
            return base_date.replace(hour=dt.hour, minute=dt.minute, second=0, microsecond=0)
            
        elif "Yesterday" in raw_time:
            # Handle "Yesterday HH:MMam/pm"
            time_part = raw_time.replace("Yesterday", "").strip()
            dt = datetime.strptime(time_part, "%I:%M%p")
            base_date = FUTURE_START if FUTURE_START > datetime.now() else datetime.now()
            return (base_date - timedelta(days=1)).replace(hour=dt.hour, minute=dt.minute, second=0, microsecond=0)
            
        elif " " in raw_time:
            # Handle "MMM-DD-YY HH:MMam/pm"
            try:
                parsed_date = datetime.strptime(raw_time, "%b-%d-%y %I:%M%p")
                # If the parsed date is before our start date, adjust it to the future
                if parsed_date < FUTURE_START:
                    years_to_add = ((FUTURE_START.year - parsed_date.year) // 1 + 1) * 1
                    return parsed_date.replace(year=parsed_date.year + years_to_add)
                return parsed_date
            except ValueError as e:
                logging.error(f"Failed to parse full date '{raw_time}': {str(e)}")
                raise
                
        else:
            # Handle time-only format "HH:MMam/pm"
            try:
                dt = datetime.strptime(raw_time, "%I:%M%p")
                base_date = FUTURE_START if FUTURE_START > datetime.now() else datetime.now()
                return base_date.replace(hour=dt.hour, minute=dt.minute, second=0, microsecond=0)
            except ValueError as e:
                logging.error(f"Failed to parse time '{raw_time}': {str(e)}")
                raise
                
    except Exception as e:
        logging.error(f"Date parsing error for '{raw_time}': {str(e)}")
        raise

def scrape_finviz_news(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    }
    
    logging.info(f"Fetching news for {ticker}")
    res = requests.get(url, headers=headers)

    if res.status_code != 200:
        logging.error(f"Failed to fetch {ticker} ({res.status_code})")
        return []

    soup = BeautifulSoup(res.text, "html.parser")
    news_table = soup.find("table", class_="fullview-news-outer")
    if not news_table:
        logging.warning(f"No news found for {ticker}")
        return []

    rows = news_table.find_all("tr")
    news = []
    current_date = None

    for row in rows:
        cols = row.find_all("td")
        if len(cols) != 2:
            continue

        date_text = cols[0].text.strip()
        headline = cols[1].text.strip()
        link = cols[1].find("a")["href"] if cols[1].find("a") else ""

        logging.debug(f"Raw date text: '{date_text}'")

        # If it's just a time (contains : but not Today/Yesterday)
        if ":" in date_text and not any(word in date_text for word in ["Today", "Yesterday"]):
            if current_date is None:
                # If we don't have a current date, use today
                current_date = "Today"
            # Use the current date with this time
            date_text = f"{current_date} {date_text}"
        else:
            # Store the full date for future time-only entries
            if "Today" in date_text or "Yesterday" in date_text:
                current_date = date_text.split()[0]  # Get just Today/Yesterday
            else:
                # For full dates like "Jul-22-25", store the whole thing
                current_date = date_text.split()[0]  # Get the date part

        try:
            timestamp = parse_finviz_time(date_text)
            if timestamp:
                # Analyze sentiment for the headline
                sentiment = analyze_sentiment(headline)
                
                news.append({
                    "ticker": ticker,
                    "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    "source": "finviz",
                    "content": {
                        "text": headline,
                        "url": link,
                        "author": "finviz"
                    },
                    "sentiment": sentiment,
                    "market_context": {}
                })
                logging.debug(f"Successfully parsed date: {date_text} -> {timestamp}")
                logging.debug(f"Sentiment for '{headline[:100]}...': {sentiment}")
        except Exception as e:
            logging.error(f"Failed to parse date '{date_text}': {str(e)}")
            continue

    return news

def save_json_news(ticker, news_data):
    if not news_data:
        logging.warning(f"No news data to save for {ticker}")
        return

    for item in news_data:
        date = datetime.strptime(item['timestamp'], '%Y-%m-%d %H:%M:%S')
        year_month = date.strftime('%Y-%m')
        
        ticker_dir = os.path.join(OUTPUT_DIR, ticker)
        output_path = os.path.join(ticker_dir, f"{year_month}.json")
        
        # Create directory if it doesn't exist
        os.makedirs(ticker_dir, exist_ok=True)
        
        # Initialize existing data
        existing = []
        
        # Try to load existing data if file exists
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            try:
                with open(output_path, 'r') as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        logging.error(f"Existing data in {output_path} is not a list. Creating new file.")
                        existing = []
            except json.JSONDecodeError as e:
                logging.error(f"Error reading {output_path}: {str(e)}. Creating new file.")
                existing = []
            
        # Avoid duplicates using both timestamp and headline
        existing_entries = {(entry['timestamp'], entry['content']['text']) for entry in existing}
        if (item['timestamp'], item['content']['text']) not in existing_entries:
            existing.append(item)
            logging.info(f"Added new entry for {ticker} at {item['timestamp']}")
            
        # Sort by timestamp
        existing.sort(key=lambda x: x['timestamp'])
        
        try:
            # Write to a temporary file first
            temp_path = output_path + '.tmp'
            with open(temp_path, 'w') as f:
                json.dump(existing, f, indent=4)
            
            # Then rename it to the actual file (atomic operation)
            os.replace(temp_path, output_path)
            logging.info(f"Successfully saved data to {output_path}")
        except Exception as e:
            logging.error(f"Error saving to {output_path}: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    logging.info(f"Processed {len(news_data)} entries for {ticker}")

def main():
    session = requests.Session()
    for ticker in TICKERS:
        retries = 3
        while retries > 0:
            try:
                logging.info(f"Starting scrape for {ticker}")
                news = scrape_finviz_news(ticker)
                if news:
                    save_json_news(ticker, news)
                else:
                    logging.warning(f"No new news for {ticker}")
                time.sleep(random.uniform(8, 15))  # More conservative rate limiting
                break
            except Exception as e:
                logging.error(f"Error with {ticker}: {str(e)}")
                retries -= 1
                time.sleep(30)  # Longer wait on error
                
        if retries == 0:
            logging.error(f"Failed to scrape {ticker} after 3 attempts")

if __name__ == "__main__":
    logging.info("Starting FinViz scraper")
    main()
    logging.info("Finished scraping")
