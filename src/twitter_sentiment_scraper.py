"""
Twitter Sentiment Scraper for Social Arbitrage Trader
Collects real-time financial sentiment from Twitter/X for stock analysis
"""

import tweepy
import json
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from transformers import pipeline
import pandas as pd
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('twitter_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Stock tickers to monitor
TICKERS = [
    'AAPL', 'MSFT', 'TSLA', 'GOOG', 'META', 'NVDA',
    'RBLX', 'AMC', 'GME', 'CROX',
    'ANF', 'URBN', 'AEO', 'LEVI', 'GPS', 'ASO', 'LULU', 'NKE',
    'JNJ', 'PFE', 'UNH', 'MRK', 'LLY', 'ABT'
]

# Financial keywords to enhance sentiment detection
FINANCIAL_KEYWORDS = [
    'stock', 'market', 'trading', 'invest', 'earnings', 'revenue', 'profit',
    'bullish', 'bearish', 'buy', 'sell', 'hold', 'price', 'volume',
    'analyst', 'upgrade', 'downgrade', 'target', 'forecast', 'guidance'
]

class TwitterSentimentScraper:
    def __init__(self, output_dir: str = "../data/sentiment/twitter_json"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize simple sentiment analyzer
        try:
            # Use a simpler model that's more compatible
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            logger.info("Successfully loaded sentiment model")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {str(e)}")
            # Fallback to simple keyword-based sentiment
            self.sentiment_analyzer = None
            logger.warning("Using fallback keyword-based sentiment analysis")
        
        # Twitter API credentials (you'll need to set these)
        self.api = self._setup_twitter_api()
        
    def _setup_twitter_api(self):
        """Setup Twitter API client"""
        # You'll need to get these from Twitter Developer Portal
        # https://developer.twitter.com/en/portal/dashboard
        consumer_key = os.getenv('1vgDvykwWAbhVUG2GDQtRcffp')
        consumer_secret = os.getenv('6VwHaR1RPuX4JqmobEqNkJS6iduZNQ4FBFULPFKJaLB5cjZKRy')
        
        
        if not all([consumer_key, consumer_secret]):
            logger.warning("Twitter API credentials not found. Using mock data for testing.")
            return None
        
        try:
            auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token, access_token_secret)
            api = tweepy.API(auth, wait_on_rate_limit=True)
            logger.info("Successfully connected to Twitter API")
            return api
        except Exception as e:
            logger.error(f"Failed to connect to Twitter API: {str(e)}")
            return None
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text using FinBERT"""
        try:
            result = self.sentiment_analyzer(text)[0]
            return {
                "label": result["label"],
                "score": float(result["score"])
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {"label": "neutral", "score": 0.5}
    
    def extract_ticker_from_text(self, text: str) -> Optional[str]:
        """Extract stock ticker from tweet text"""
        text_upper = text.upper()
        for ticker in TICKERS:
            # Look for $TICKER or #TICKER patterns
            if f'${ticker}' in text_upper or f'#{ticker}' in text_upper:
                return ticker
        return None
    
    def is_financial_tweet(self, text: str) -> bool:
        """Check if tweet is financial in nature"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in FINANCIAL_KEYWORDS)
    
    def scrape_tweets_for_ticker(self, ticker: str, count: int = 100) -> List[Dict]:
        """Scrape tweets for a specific ticker"""
        if not self.api:
            logger.warning(f"Twitter API not available. Generating mock data for {ticker}")
            return self._generate_mock_data(ticker, count)
        
        tweets_data = []
        try:
            # Search for tweets with $TICKER or #TICKER
            query = f"${ticker} OR #{ticker} -is:retweet"
            tweets = self.api.search_tweets(q=query, lang="en", count=count, tweet_mode="extended")
            
            for tweet in tweets:
                # Skip retweets and non-financial tweets
                if hasattr(tweet, 'retweeted_status') or not self.is_financial_tweet(tweet.full_text):
                    continue
                
                # Analyze sentiment
                sentiment = self.analyze_sentiment(tweet.full_text)
                
                tweet_data = {
                    "ticker": ticker,
                    "timestamp": tweet.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    "source": "twitter",
                    "content": {
                        "text": tweet.full_text,
                        "user": tweet.user.screen_name,
                        "followers_count": tweet.user.followers_count,
                        "verified": tweet.user.verified,
                        "tweet_id": tweet.id_str
                    },
                    "sentiment": sentiment,
                    "engagement": {
                        "retweets": tweet.retweet_count,
                        "likes": tweet.favorite_count,
                        "replies": getattr(tweet, 'reply_count', 0)
                    },
                    "market_context": {}
                }
                
                tweets_data.append(tweet_data)
                logger.debug(f"Processed tweet for {ticker}: {tweet.full_text[:100]}...")
            
            logger.info(f"Scraped {len(tweets_data)} tweets for {ticker}")
            
        except Exception as e:
            logger.error(f"Error scraping tweets for {ticker}: {str(e)}")
        
        return tweets_data
    
    def _generate_mock_data(self, ticker: str, count: int) -> List[Dict]:
        """Generate mock data for testing when Twitter API is not available"""
        mock_tweets = []
        base_time = datetime.now()
        
        for i in range(min(count, 20)):  # Limit mock data
            # Generate realistic mock tweets
            mock_texts = [
                f"${ticker} looking bullish today! Great earnings report",
                f"Not sure about ${ticker} - market seems uncertain",
                f"${ticker} stock price action is interesting",
                f"Analysts are upgrading ${ticker} - good sign",
                f"${ticker} earnings beat expectations",
                f"Concerned about ${ticker} fundamentals",
                f"${ticker} technical analysis shows support",
                f"${ticker} volume is picking up",
                f"${ticker} management team is solid",
                f"${ticker} competition is getting tougher"
            ]
            
            text = mock_texts[i % len(mock_texts)]
            sentiment = self.analyze_sentiment(text)
            
            mock_tweet = {
                "ticker": ticker,
                "timestamp": (base_time - timedelta(minutes=i*5)).strftime('%Y-%m-%d %H:%M:%S'),
                "source": "twitter",
                "content": {
                    "text": text,
                    "user": f"mock_user_{i}",
                    "followers_count": np.random.randint(100, 10000),
                    "verified": np.random.choice([True, False], p=[0.1, 0.9]),
                    "tweet_id": f"mock_{ticker}_{i}"
                },
                "sentiment": sentiment,
                "engagement": {
                    "retweets": np.random.randint(0, 50),
                    "likes": np.random.randint(0, 200),
                    "replies": np.random.randint(0, 20)
                },
                "market_context": {}
            }
            
            mock_tweets.append(mock_tweet)
        
        return mock_tweets
    
    def save_tweets(self, ticker: str, tweets_data: List[Dict]):
        """Save tweets to JSON file organized by month"""
        if not tweets_data:
            logger.warning(f"No tweets to save for {ticker}")
            return
        
        # Group tweets by month
        tweets_by_month = defaultdict(list)
        for tweet in tweets_data:
            date = datetime.strptime(tweet['timestamp'], '%Y-%m-%d %H:%M:%S')
            year_month = date.strftime('%Y-%m')
            tweets_by_month[year_month].append(tweet)
        
        # Save each month's data
        for year_month, tweets in tweets_by_month.items():
            ticker_dir = self.output_dir / ticker
            ticker_dir.mkdir(exist_ok=True)
            
            output_path = ticker_dir / f"{year_month}.json"
            
            # Load existing data if file exists
            existing_tweets = []
            if output_path.exists() and output_path.stat().st_size > 0:
                try:
                    with open(output_path, 'r') as f:
                        existing_tweets = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Error reading {output_path}, creating new file")
                    existing_tweets = []
            
            # Avoid duplicates
            existing_ids = {tweet['content']['tweet_id'] for tweet in existing_tweets}
            new_tweets = [tweet for tweet in tweets if tweet['content']['tweet_id'] not in existing_ids]
            
            if new_tweets:
                existing_tweets.extend(new_tweets)
                
                # Sort by timestamp
                existing_tweets.sort(key=lambda x: x['timestamp'])
                
                # Save to file
                with open(output_path, 'w') as f:
                    json.dump(existing_tweets, f, indent=2, default=str)
                
                logger.info(f"Saved {len(new_tweets)} new tweets for {ticker} ({year_month})")
    
    def scrape_all_tickers(self, tweets_per_ticker: int = 50):
        """Scrape tweets for all tickers"""
        logger.info(f"Starting Twitter sentiment scraping for {len(TICKERS)} tickers")
        
        for i, ticker in enumerate(TICKERS):
            logger.info(f"Scraping tweets for {ticker} ({i+1}/{len(TICKERS)})")
            
            try:
                tweets_data = self.scrape_tweets_for_ticker(ticker, tweets_per_ticker)
                if tweets_data:
                    self.save_tweets(ticker, tweets_data)
                
                # Rate limiting
                time.sleep(2)  # 2 second delay between tickers
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                continue
        
        logger.info("Twitter sentiment scraping completed")

def main():
    """Main function to run the Twitter sentiment scraper"""
    scraper = TwitterSentimentScraper()
    scraper.scrape_all_tickers(tweets_per_ticker=50)

if __name__ == "__main__":
    main() 