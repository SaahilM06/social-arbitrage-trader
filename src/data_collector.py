import os
import praw
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv



load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

def scrape_reddit_data(tickers, limit = 100, subreddit_name = 'wallstreetbets'):
    posts = []
    subreddit = reddit.subreddit(subreddit_name)

    for ticker in tickers:
        print(f"Scraping '{ticker}' posts from r/{subreddit_name}...")
        for submission in subreddit.search(ticker, sort='new', limit=limit):
            posts.append({
                'platform': 'reddit',
                'subreddit': subreddit_name,
                'ticker': ticker,
                'title': submission.title,
                'body': submission.selftext,
                'created_utc': datetime.fromtimestamp(submission.created_utc),
                'score': submission.score,
                'num_comments': submission.num_comments,
                'url': submission.url
            })

    df = pd.DataFrame(posts)
    return df

def save_reddit_data(tickers, limit = 100):
    df = scrape_reddit_data(tickers, limit = limit)
    if not df.empty:
        output_path = '/Users/saahi/social-arbitrage-trader/data/processed/reddit_posts.csv'
        df.to_csv(output_path, index=False)
        print("saved to csv")


if __name__ == "__main__":
    tickers_to_scrape = ['AAPL', 'TSLA', 'NVDA']
    save_reddit_data(tickers_to_scrape, limit=100)



