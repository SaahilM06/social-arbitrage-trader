"""
Reddit Sentiment Scraper for Social Arbitrage Trader
Collects financial sentiment from Reddit communities for stock analysis
"""

import praw
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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_scraper.log'),
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

# Financial subreddits to scrape
FINANCIAL_SUBREDDITS = [
    'wallstreetbets', 'stocks', 'investing', 'StockMarket',
    'pennystocks', 'options', 'daytrading', 'algotrading'
]

# Stock-specific subreddits
STOCK_SUBREDDITS = {
    'AAPL': ['AAPL', 'apple'],
    'MSFT': ['MSFT', 'microsoft'],
    'TSLA': ['teslamotors', 'teslainvestorsclub'],
    'GOOG': ['google', 'alphabet'],
    'META': ['facebook', 'meta'],
    'NVDA': ['NVIDIA', 'nvidia'],
    'GME': ['GME', 'Superstonk'],
    'AMC': ['AMC', 'amcstock']
}

class RedditSentimentScraper:
    def __init__(self, output_dir: str = "../data/sentiment/reddit_json"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sentiment analyzer
        try:
            # Use a model that doesn't have torch.load vulnerability issues
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            logger.info("Successfully loaded sentiment model")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {str(e)}")
            raise
        
        # Reddit API setup
        self.reddit = self._setup_reddit_api()
        
    def _setup_reddit_api(self):
        """Setup Reddit API client"""
        # You'll need to get these from Reddit Developer Portal
        # https://www.reddit.com/prefs/apps
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        user_agent = os.getenv('REDDIT_USER_AGENT', 'SocialArbitrageTrader/1.0')
        
        if not all([client_id, client_secret]):
            logger.warning("Reddit API credentials not found. Using mock data for testing.")
            return None
        
        try:
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            logger.info("Successfully connected to Reddit API")
            return reddit
        except Exception as e:
            logger.error(f"Failed to connect to Reddit API: {str(e)}")
            return None
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text using FinBERT"""
        try:
            # Truncate text if too long for FinBERT
            if len(text) > 500:
                text = text[:500]
            
            result = self.sentiment_analyzer(text)[0]
            return {
                "label": result["label"],
                "score": float(result["score"])
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {"label": "neutral", "score": 0.5}
    
    def extract_ticker_from_text(self, text: str) -> Optional[str]:
        """Extract stock ticker from text"""
        text_upper = text.upper()
        for ticker in TICKERS:
            # Look for $TICKER, #TICKER, or just TICKER patterns
            if (f'${ticker}' in text_upper or 
                f'#{ticker}' in text_upper or 
                f' {ticker} ' in text_upper or
                text_upper.startswith(f'{ticker} ') or
                text_upper.endswith(f' {ticker}')):
                return ticker
        return None
    
    def is_financial_content(self, text: str) -> bool:
        """Check if content is financial in nature"""
        text_lower = text.lower()
        financial_keywords = [
            'stock', 'market', 'trading', 'invest', 'earnings', 'revenue', 'profit',
            'bullish', 'bearish', 'buy', 'sell', 'hold', 'price', 'volume',
            'analyst', 'upgrade', 'downgrade', 'target', 'forecast', 'guidance',
            'portfolio', 'position', 'call', 'put', 'option', 'dividend'
        ]
        return any(keyword in text_lower for keyword in financial_keywords)
    
    def scrape_subreddit_for_ticker(self, ticker: str, subreddit_name: str, 
                                   limit: int = 100, time_filter: str = 'month') -> List[Dict]:
        """Scrape posts from a specific subreddit for a ticker"""
        if not self.reddit:
            logger.warning(f"Reddit API not available. Generating mock data for {ticker}")
            return self._generate_mock_data(ticker, limit)
        
        posts_data = []
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Search for posts containing the ticker
            search_query = f"{ticker} OR ${ticker}"
            posts = subreddit.search(search_query, limit=limit, time_filter=time_filter)
            
            for post in posts:
                # Check if post is about the specific ticker
                if not self._is_post_about_ticker(post, ticker):
                    continue
                
                # Analyze sentiment of post title and body
                title_sentiment = self.analyze_sentiment(post.title)
                body_sentiment = self.analyze_sentiment(post.selftext) if post.selftext else title_sentiment
                
                # Use the more confident sentiment
                sentiment = title_sentiment if title_sentiment['score'] > body_sentiment['score'] else body_sentiment
                
                post_data = {
                    "ticker": ticker,
                    "timestamp": datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                    "source": "reddit",
                    "subreddit": subreddit_name,
                    "content": {
                        "title": post.title,
                        "body": post.selftext,
                        "author": str(post.author) if post.author else "deleted",
                        "post_id": post.id,
                        "url": f"https://reddit.com{post.permalink}"
                    },
                    "sentiment": sentiment,
                    "engagement": {
                        "upvotes": post.score,
                        "comments": post.num_comments,
                        "upvote_ratio": post.upvote_ratio
                    },
                    "market_context": {}
                }
                
                posts_data.append(post_data)
                logger.debug(f"Processed post for {ticker}: {post.title[:100]}...")
            
            logger.info(f"Scraped {len(posts_data)} posts for {ticker} from r/{subreddit_name}")
            
        except Exception as e:
            logger.error(f"Error scraping r/{subreddit_name} for {ticker}: {str(e)}")
        
        return posts_data
    
    def _is_post_about_ticker(self, post, ticker: str) -> bool:
        """Check if post is specifically about the ticker"""
        text = f"{post.title} {post.selftext}".upper()
        ticker_upper = ticker.upper()
        
        # Look for ticker mentions
        ticker_patterns = [f'${ticker_upper}', f'#{ticker_upper}', f' {ticker_upper} ']
        return any(pattern in text for pattern in ticker_patterns)
    
    def _generate_mock_data(self, ticker: str, count: int) -> List[Dict]:
        """Generate mock data for testing when Reddit API is not available"""
        mock_posts = []
        base_time = datetime.now()
        
        for i in range(min(count, 15)):  # Limit mock data
            # Generate realistic mock posts
            mock_titles = [
                f"${ticker} DD - Why I'm bullish on this stock",
                f"${ticker} earnings analysis - what to expect",
                f"Thoughts on ${ticker} current price action?",
                f"${ticker} technical analysis - support and resistance",
                f"${ticker} vs competitors - fundamental analysis",
                f"${ticker} options strategy discussion",
                f"${ticker} dividend analysis and yield",
                f"${ticker} management team review",
                f"${ticker} future growth prospects",
                f"${ticker} risk assessment and concerns"
            ]
            
            mock_bodies = [
                f"Here's my analysis of ${ticker}. The fundamentals look strong...",
                f"I've been following ${ticker} for a while now. Here's what I think...",
                f"Looking at ${ticker} technical indicators, I see potential...",
                f"${ticker} has been on my radar. Let me share my thoughts...",
                f"After researching ${ticker}, here's my investment thesis..."
            ]
            
            title = mock_titles[i % len(mock_titles)]
            body = mock_bodies[i % len(mock_bodies)]
            sentiment = self.analyze_sentiment(f"{title} {body}")
            
            mock_post = {
                "ticker": ticker,
                "timestamp": (base_time - timedelta(days=i)).strftime('%Y-%m-%d %H:%M:%S'),
                "source": "reddit",
                "subreddit": "stocks",
                "content": {
                    "title": title,
                    "body": body,
                    "author": f"mock_user_{i}",
                    "post_id": f"mock_{ticker}_{i}",
                    "url": f"https://reddit.com/r/stocks/mock_{ticker}_{i}"
                },
                "sentiment": sentiment,
                "engagement": {
                    "upvotes": np.random.randint(0, 500),
                    "comments": np.random.randint(0, 100),
                    "upvote_ratio": np.random.uniform(0.5, 1.0)
                },
                "market_context": {}
            }
            
            mock_posts.append(mock_post)
        
        return mock_posts
    
    def save_posts(self, ticker: str, posts_data: List[Dict]):
        """Save posts to JSON file organized by month"""
        if not posts_data:
            logger.warning(f"No posts to save for {ticker}")
            return
        
        # Group posts by month
        posts_by_month = defaultdict(list)
        for post in posts_data:
            date = datetime.strptime(post['timestamp'], '%Y-%m-%d %H:%M:%S')
            year_month = date.strftime('%Y-%m')
            posts_by_month[year_month].append(post)
        
        # Save each month's data
        for year_month, posts in posts_by_month.items():
            ticker_dir = self.output_dir / ticker
            ticker_dir.mkdir(exist_ok=True)
            
            output_path = ticker_dir / f"{year_month}.json"
            
            # Load existing data if file exists
            existing_posts = []
            if output_path.exists() and output_path.stat().st_size > 0:
                try:
                    with open(output_path, 'r') as f:
                        existing_posts = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Error reading {output_path}, creating new file")
                    existing_posts = []
            
            # Avoid duplicates
            existing_ids = {post['content']['post_id'] for post in existing_posts}
            new_posts = [post for post in posts if post['content']['post_id'] not in existing_ids]
            
            if new_posts:
                existing_posts.extend(new_posts)
                
                # Sort by timestamp
                existing_posts.sort(key=lambda x: x['timestamp'])
                
                # Save to file
                with open(output_path, 'w') as f:
                    json.dump(existing_posts, f, indent=2)
                
                logger.info(f"Saved {len(new_posts)} new posts for {ticker} ({year_month})")
    
    def scrape_all_tickers(self, posts_per_ticker: int = 100, start_date: str = "2023-04-01"):
        """Scrape posts for all tickers from April 2023 to now"""
        logger.info(f"Starting Reddit sentiment scraping for {len(TICKERS)} tickers from {start_date}")
        
        # Time filters to try (Reddit API limitations)
        time_filters = ['year', 'month', 'week', 'day']
        
        for i, ticker in enumerate(TICKERS):
            logger.info(f"Scraping posts for {ticker} ({i+1}/{len(TICKERS)})")
            
            all_posts = []
            
            # Try different time filters to get more historical data
            for time_filter in time_filters:
                logger.info(f"  Trying time filter: {time_filter}")
                
                # Scrape from general financial subreddits
                for subreddit in FINANCIAL_SUBREDDITS[:4]:  # Use top 4 subreddits
                    try:
                        posts = self.scrape_subreddit_for_ticker(ticker, subreddit, 
                                                               posts_per_ticker//4, time_filter)
                        all_posts.extend(posts)
                        time.sleep(1)  # Rate limiting
                    except Exception as e:
                        logger.error(f"Error scraping r/{subreddit} for {ticker}: {str(e)}")
                        continue
                
                # Scrape from stock-specific subreddits if available
                if ticker in STOCK_SUBREDDITS:
                    for subreddit in STOCK_SUBREDDITS[ticker]:
                        try:
                            posts = self.scrape_subreddit_for_ticker(ticker, subreddit, 
                                                                   posts_per_ticker//2, time_filter)
                            all_posts.extend(posts)
                            time.sleep(1)  # Rate limiting
                        except Exception as e:
                            logger.error(f"Error scraping r/{subreddit} for {ticker}: {str(e)}")
                            continue
                
                # If we got enough posts, break early
                if len(all_posts) >= posts_per_ticker * 2:
                    break
                
                time.sleep(2)  # Rate limiting between time filters
            
            # Filter posts by date (keep only posts from April 2023 onwards)
            start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            filtered_posts = []
            for post in all_posts:
                post_date = datetime.strptime(post['timestamp'], '%Y-%m-%d %H:%M:%S')
                if post_date >= start_datetime:
                    filtered_posts.append(post)
            
            logger.info(f"  Found {len(filtered_posts)} posts from {start_date} onwards for {ticker}")
            
            if filtered_posts:
                self.save_posts(ticker, filtered_posts)
            
            # Rate limiting between tickers
            time.sleep(3)
        
        logger.info("Reddit sentiment scraping completed")

def main():
    """Main function to run the Reddit sentiment scraper"""
    scraper = RedditSentimentScraper()
    scraper.scrape_all_tickers(posts_per_ticker=100, start_date="2025-04-23")

if __name__ == "__main__":
    main() 