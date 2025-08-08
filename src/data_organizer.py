"""
Data Organizer for Social Arbitrage Trader
Consolidates and structures data from multiple sources for analysis
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataOrganizer:
    def __init__(self, base_path: str = "data"):
        # Handle both relative and absolute paths
        if Path(base_path).is_absolute():
            self.base_path = Path(base_path)
        else:
            # If running from src directory, go up one level
            current_dir = Path.cwd()
            if current_dir.name == "src":
                self.base_path = current_dir.parent / base_path
            else:
                self.base_path = Path(base_path)
        self.organized_path = self.base_path / "organized"
        self.organized_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.organized_path / "market_data").mkdir(exist_ok=True)
        (self.organized_path / "sentiment_data").mkdir(exist_ok=True)
        (self.organized_path / "news_data").mkdir(exist_ok=True)
        (self.organized_path / "combined_data").mkdir(exist_ok=True)
        
    def load_ohlcv_data(self, ticker: str) -> pd.DataFrame:
        """Load and structure OHLCV data"""
        file_path = self.base_path / "processed" / f"{ticker}_ohlcv.json"
        
        logger.info(f"Looking for OHLCV data at: {file_path}")
        if not file_path.exists():
            logger.warning(f"No OHLCV data found for {ticker} at {file_path}")
            return pd.DataFrame()
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        logger.info(f"Loaded {len(data)} data points for {ticker}")
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data, orient='index')
        logger.info(f"DataFrame shape: {df.shape}, Index type: {type(df.index)}")
        
        # Convert index to datetime with error handling
        try:
            df.index = pd.to_datetime(df.index)
            logger.info(f"Successfully converted index to datetime. Index type: {type(df.index)}")
        except Exception as e:
            logger.error(f"Failed to convert index to datetime for {ticker}: {e}")
            return pd.DataFrame()
            
        df = df.sort_index()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error(f"Index is not DatetimeIndex for {ticker}: {type(df.index)}")
            return pd.DataFrame()
        
        # Add ticker column
        df['ticker'] = ticker
        
        return df
    
    def load_news_data(self, ticker: str) -> pd.DataFrame:
        """Load and structure news data"""
        file_path = self.base_path / "news" / f"{ticker}.json"
        
        if not file_path.exists():
            logger.warning(f"No news data found for {ticker}")
            return pd.DataFrame()
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['published_utc'] = pd.to_datetime(df['published_utc'])
        df = df.sort_values('published_utc')
        
        # Extract sentiment from insights
        df['sentiment'] = df['insights'].apply(
            lambda x: next((insight['sentiment'] for insight in x if insight['ticker'] == ticker), 'neutral')
        )
        df['sentiment_reasoning'] = df['insights'].apply(
            lambda x: next((insight['sentiment_reasoning'] for insight in x if insight['ticker'] == ticker), '')
        )
        
        # Add ticker column
        df['ticker'] = ticker
        
        return df
    
    def load_sentiment_data(self, ticker: str) -> pd.DataFrame:
        """Load and structure Finviz sentiment data"""
        sentiment_path = self.base_path / "sentiment" / "finviz_json" / ticker
        
        if not sentiment_path.exists():
            logger.warning(f"No Finviz sentiment data found for {ticker}")
            return pd.DataFrame()
            
        all_data = []
        
        # Load all monthly files
        for file_path in sentiment_path.glob("*.json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_data.extend(data)
        
        if not all_data:
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Extract sentiment scores
        df['sentiment_label'] = df['sentiment'].apply(lambda x: x.get('label', 'neutral'))
        df['sentiment_score'] = df['sentiment'].apply(lambda x: x.get('score', 0.5))
        
        # Add ticker column
        df['ticker'] = ticker
        
        return df
    
    def load_reddit_sentiment_data(self, ticker: str) -> pd.DataFrame:
        """Load and structure Reddit sentiment data"""
        reddit_file = self.base_path / "sentiment" / "reddit_combined" / f"{ticker}_combined.json"
        
        if not reddit_file.exists():
            logger.warning(f"No Reddit sentiment data found for {ticker}")
            return pd.DataFrame()
            
        with open(reddit_file, 'r') as f:
            data = json.load(f)
        
        if not data:
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Extract sentiment scores
        df['sentiment_label'] = df['sentiment'].apply(lambda x: x.get('label', 'neutral'))
        df['sentiment_score'] = df['sentiment'].apply(lambda x: x.get('score', 0.5))
        
        # Add ticker column
        df['ticker'] = ticker
        
        return df
    
    def create_time_aligned_dataset(self, ticker: str) -> pd.DataFrame:
        """Create time-aligned dataset with market data, news, and sentiment"""
        logger.info(f"Creating time-aligned dataset for {ticker}")
        
        # Load all data sources
        market_df = self.load_ohlcv_data(ticker)
        news_df = self.load_news_data(ticker)
        finviz_sentiment_df = self.load_sentiment_data(ticker)
        reddit_sentiment_df = self.load_reddit_sentiment_data(ticker)
        
        if market_df.empty:
            logger.error(f"No market data available for {ticker}")
            return pd.DataFrame()
        
        # Start with market data as base
        result_df = market_df.copy()
        
        # Add news sentiment features
        if not news_df.empty:
            # Convert news timestamps to same timezone as market data
            news_df['published_utc'] = news_df['published_utc'].dt.tz_localize(None)
            
            # Aggregate news sentiment by 5-minute intervals to match OHLCV data
            news_df['time_interval'] = news_df['published_utc'].dt.floor('5min')
            
            # Create news sentiment breakdown with ML-optimized features
            def get_news_breakdown(group):
                positive_count = (group['sentiment'] == 'positive').sum()
                negative_count = (group['sentiment'] == 'negative').sum()
                neutral_count = (group['sentiment'] == 'neutral').sum()
                total_count = len(group)
                
                # ML-optimized features
                news_sentiment_ratio = positive_count / (total_count + 1e-8)  # Avoid division by zero
                sentiment_intensity = abs(positive_count - negative_count) / (total_count + 1e-8)
                news_volume_score = np.log(total_count + 1)  # Handle sparse data
                sentiment_momentum = (positive_count - negative_count) * total_count  # Weighted sentiment
                
                # Categorical features
                dominant_sentiment = 'neutral'
                if positive_count > negative_count and positive_count > neutral_count:
                    dominant_sentiment = 'positive'
                elif negative_count > positive_count and negative_count > neutral_count:
                    dominant_sentiment = 'negative'
                
                sentiment_consensus = 1 if len(set(group['sentiment'])) == 1 else 0  # All same sentiment
                high_impact_news = 1 if total_count > 2 else 0  # Multiple news items
                
                return pd.Series({
                    'news_sentiment_score': positive_count - negative_count,  # Net sentiment
                    'news_count': total_count,
                    'news_sentiment_ratio': news_sentiment_ratio,
                    'news_sentiment_intensity': sentiment_intensity,
                    'news_volume_score': news_volume_score,
                    'news_sentiment_momentum': sentiment_momentum,
                    'news_dominant_sentiment': dominant_sentiment,
                    'news_sentiment_consensus': sentiment_consensus,
                    'high_impact_news': high_impact_news,
                    'news_positive_count': positive_count,
                    'news_negative_count': negative_count,
                    'news_neutral_count': neutral_count
                })
            
            interval_news = news_df.groupby('time_interval').apply(get_news_breakdown)
            
            # Merge with market data using join to preserve index
            result_df['time_interval'] = result_df.index.floor('5min')
            result_df = result_df.join(interval_news, on='time_interval', how='left')
            result_df = result_df.drop('time_interval', axis=1)
            result_df = result_df.fillna(0)
        
        # Add Finviz sentiment features (supplementary - only available July-August)
        if not finviz_sentiment_df.empty:
            # Convert sentiment timestamps to same timezone as market data
            finviz_sentiment_df['timestamp'] = finviz_sentiment_df['timestamp'].dt.tz_localize(None)
            
            # Aggregate sentiment by 5-minute intervals to match OHLCV data
            finviz_sentiment_df['time_interval'] = finviz_sentiment_df['timestamp'].dt.floor('5min')
            
            # Create sentiment breakdown with ML-optimized features
            def get_sentiment_breakdown(group):
                positive_count = (group['sentiment_label'] == 'positive').sum()
                negative_count = (group['sentiment_label'] == 'negative').sum()
                neutral_count = (group['sentiment_label'] == 'neutral').sum()
                total_count = len(group)
                
                # ML-optimized features
                sentiment_ratio = positive_count / (total_count + 1e-8)  # Avoid division by zero
                sentiment_intensity = abs(positive_count - negative_count) / (total_count + 1e-8)
                sentiment_volume_score = np.log(total_count + 1)  # Handle sparse data
                sentiment_momentum = (positive_count - negative_count) * total_count  # Weighted sentiment
                avg_sentiment_score = group['sentiment_score'].mean()
                
                # Categorical features
                dominant_sentiment = 'neutral'
                if positive_count > negative_count and positive_count > neutral_count:
                    dominant_sentiment = 'positive'
                elif negative_count > positive_count and negative_count > neutral_count:
                    dominant_sentiment = 'negative'
                
                sentiment_consensus = 1 if len(set(group['sentiment_label'])) == 1 else 0  # All same sentiment
                high_impact_sentiment = 1 if total_count > 5 else 0  # Multiple sentiment items
                
                return pd.Series({
                    'finviz_sentiment_score': avg_sentiment_score,
                    'finviz_sentiment_ratio': sentiment_ratio,
                    'finviz_sentiment_intensity': sentiment_intensity,
                    'finviz_sentiment_volume_score': sentiment_volume_score,
                    'finviz_sentiment_momentum': sentiment_momentum,
                    'finviz_dominant_sentiment': dominant_sentiment,
                    'finviz_sentiment_consensus': sentiment_consensus,
                    'high_impact_finviz_sentiment': high_impact_sentiment,
                    'finviz_positive_count': positive_count,
                    'finviz_negative_count': negative_count,
                    'finviz_neutral_count': neutral_count,
                    'total_finviz_sentiment_items': total_count
                })
            
            interval_sentiment = finviz_sentiment_df.groupby('time_interval').apply(get_sentiment_breakdown)
            
            # Merge with market data using join to preserve index
            result_df['time_interval'] = result_df.index.floor('5min')
            result_df = result_df.join(interval_sentiment, on='time_interval', how='left')
            result_df = result_df.drop('time_interval', axis=1)
            result_df = result_df.fillna(0)
        else:
            # Add placeholder columns for Finviz sentiment (will be 0 for April-June)
            finviz_columns = [
                'finviz_sentiment_score', 'finviz_sentiment_ratio', 'finviz_sentiment_intensity',
                'finviz_sentiment_volume_score', 'finviz_sentiment_momentum', 'finviz_dominant_sentiment',
                'finviz_sentiment_consensus', 'high_impact_finviz_sentiment', 'finviz_positive_count',
                'finviz_negative_count', 'finviz_neutral_count', 'total_finviz_sentiment_items'
            ]
            for col in finviz_columns:
                result_df[col] = 0
        
        # Process Reddit sentiment data
        if not reddit_sentiment_df.empty:
            # Convert Reddit timestamps to same timezone as market data
            reddit_sentiment_df['timestamp'] = reddit_sentiment_df['timestamp'].dt.tz_localize(None)
            
            # Aggregate Reddit sentiment by 5-minute intervals to match OHLCV data
            reddit_sentiment_df['time_interval'] = reddit_sentiment_df['timestamp'].dt.floor('5min')
            
            # Create Reddit sentiment breakdown with ML-optimized features
            def get_reddit_breakdown(group):
                positive_count = (group['sentiment_label'] == 'positive').sum()
                negative_count = (group['sentiment_label'] == 'negative').sum()
                neutral_count = (group['sentiment_label'] == 'neutral').sum()
                total_count = len(group)
                
                # ML-optimized features
                sentiment_ratio = positive_count / (total_count + 1e-8)  # Avoid division by zero
                sentiment_intensity = abs(positive_count - negative_count) / (total_count + 1e-8)
                sentiment_volume_score = np.log(total_count + 1)  # Handle sparse data
                sentiment_momentum = (positive_count - negative_count) * total_count  # Weighted sentiment
                avg_sentiment_score = group['sentiment_score'].mean()
                
                # Categorical features
                dominant_sentiment = 'neutral'
                if positive_count > negative_count and positive_count > neutral_count:
                    dominant_sentiment = 'positive'
                elif negative_count > positive_count and negative_count > neutral_count:
                    dominant_sentiment = 'negative'
                
                sentiment_consensus = 1 if len(set(group['sentiment_label'])) == 1 else 0  # All same sentiment
                high_impact_sentiment = 1 if total_count > 3 else 0  # Multiple sentiment items
                
                return pd.Series({
                    'reddit_sentiment_score': avg_sentiment_score,
                    'reddit_sentiment_ratio': sentiment_ratio,
                    'reddit_sentiment_intensity': sentiment_intensity,
                    'reddit_sentiment_volume_score': sentiment_volume_score,
                    'reddit_sentiment_momentum': sentiment_momentum,
                    'reddit_dominant_sentiment': dominant_sentiment,
                    'reddit_sentiment_consensus': sentiment_consensus,
                    'high_impact_reddit_sentiment': high_impact_sentiment,
                    'reddit_positive_count': positive_count,
                    'reddit_negative_count': negative_count,
                    'reddit_neutral_count': neutral_count,
                    'total_reddit_sentiment_items': total_count
                })
            
            interval_reddit = reddit_sentiment_df.groupby('time_interval').apply(get_reddit_breakdown)
            
            # Merge with market data using join to preserve index
            result_df['time_interval'] = result_df.index.floor('5min')
            result_df = result_df.join(interval_reddit, on='time_interval', how='left')
            result_df = result_df.drop('time_interval', axis=1)
            result_df = result_df.fillna(0)
        else:
            # Add placeholder columns for Reddit sentiment
            reddit_columns = [
                'reddit_sentiment_score', 'reddit_sentiment_ratio', 'reddit_sentiment_intensity',
                'reddit_sentiment_volume_score', 'reddit_sentiment_momentum', 'reddit_dominant_sentiment',
                'reddit_sentiment_consensus', 'high_impact_reddit_sentiment', 'reddit_positive_count',
                'reddit_negative_count', 'reddit_neutral_count', 'total_reddit_sentiment_items'
            ]
            for col in reddit_columns:
                result_df[col] = 0
        
        # Add technical indicators
        result_df['price_change'] = result_df['close'].pct_change()
        result_df['volume_change'] = result_df['volume'].pct_change()
        result_df['volatility'] = result_df['price_change'].rolling(12).std()  # 1-hour volatility
        
        # Add time features
        result_df['hour_of_day'] = result_df.index.hour
        result_df['day_of_week'] = result_df.index.dayofweek
        result_df['is_market_open'] = ((result_df.index.hour >= 9) & (result_df.index.hour < 16)).astype(int)
        
        # Add ML-optimized target variables
        result_df['target_5min_return'] = result_df['close'].shift(-1) / result_df['close'] - 1  # 5-minute future return
        result_df['target_15min_return'] = result_df['close'].shift(-3) / result_df['close'] - 1  # 15-minute future return
        result_df['target_direction_5min'] = (result_df['target_5min_return'] > 0).astype(int)  # Binary direction
        result_df['target_direction_15min'] = (result_df['target_15min_return'] > 0).astype(int)  # Binary direction
        
        # Add lagged features for time series modeling
        if 'news_sentiment_score' in result_df.columns:
            result_df['news_sentiment_lag_1'] = result_df['news_sentiment_score'].shift(1)
            result_df['news_sentiment_lag_5'] = result_df['news_sentiment_score'].shift(5)
            result_df['news_sentiment_ma_1h'] = result_df['news_sentiment_score'].rolling(12).mean()  # 1-hour moving average
        
        if 'sentiment_score' in result_df.columns:
            result_df['sentiment_score_lag_1'] = result_df['sentiment_score'].shift(1)
            result_df['sentiment_score_lag_5'] = result_df['sentiment_score'].shift(5)
            result_df['sentiment_score_ma_1h'] = result_df['sentiment_score'].rolling(12).mean()  # 1-hour moving average
        
        # Add Reddit sentiment lagged features
        if 'reddit_sentiment_score' in result_df.columns:
            result_df['reddit_sentiment_lag_1'] = result_df['reddit_sentiment_score'].shift(1)
            result_df['reddit_sentiment_lag_5'] = result_df['reddit_sentiment_score'].shift(5)
            result_df['reddit_sentiment_ma_1h'] = result_df['reddit_sentiment_score'].rolling(12).mean()  # 1-hour moving average
        
        # Add price momentum features
        result_df['price_momentum_5min'] = result_df['close'] / result_df['close'].shift(1) - 1
        result_df['price_momentum_15min'] = result_df['close'] / result_df['close'].shift(3) - 1
        result_df['volume_momentum_5min'] = result_df['volume'] / result_df['volume'].shift(1) - 1
        
        return result_df
    
    def create_summary_statistics(self, ticker: str) -> Dict:
        """Create summary statistics for a ticker"""
        df = self.create_time_aligned_dataset(ticker)
        
        if df.empty:
            return {}
        
        summary = {
            'ticker': ticker,
            'data_start': df.index.min().isoformat(),
            'data_end': df.index.max().isoformat(),
            'total_periods': len(df),
            'market_data_points': len(df),
            'avg_price': df['close'].mean(),
            'price_volatility': df['close'].std(),
            'total_volume': df['volume'].sum(),
            'avg_volume': df['volume'].mean(),
            'max_price': df['close'].max(),
            'min_price': df['close'].min(),
            'price_range': df['close'].max() - df['close'].min(),
            'positive_sentiment_ratio': (df['news_sentiment_score'] > 0).mean() if 'news_sentiment_score' in df.columns else 0,
            'negative_sentiment_ratio': (df['news_sentiment_score'] < 0).mean() if 'news_sentiment_score' in df.columns else 0,
        }
        
        return summary
    
    def organize_all_data(self, tickers: List[str]) -> None:
        """Organize data for all tickers"""
        logger.info(f"Organizing data for {len(tickers)} tickers")
        
        all_summaries = []
        
        for ticker in tickers:
            logger.info(f"Processing {ticker}...")
            
            # Create time-aligned dataset
            df = self.create_time_aligned_dataset(ticker)
            
            if not df.empty:
                # Save organized data as JSON
                output_path = self.organized_path / "combined_data" / f"{ticker}_organized.json"
                # Convert DataFrame to JSON with datetime index as string
                # Replace NaN values with null for valid JSON
                df_clean = df.replace([np.inf, -np.inf], np.nan)
                # Convert to dict and handle NaN values
                json_data = df_clean.reset_index().to_dict('records')
                # Replace NaN values with None in the JSON data
                for record in json_data:
                    for key, value in record.items():
                        # Handle different types of NaN values
                        if isinstance(value, (list, np.ndarray)):
                            # Skip arrays/lists - they might contain valid data
                            continue
                        elif pd.isna(value):
                            record[key] = None
                with open(output_path, 'w') as f:
                    json.dump(json_data, f, indent=2, default=str)
                logger.info(f"Saved organized data for {ticker} to {output_path}")
                
                # Create summary
                summary = self.create_summary_statistics(ticker)
                all_summaries.append(summary)
        
        # Save overall summary
        if all_summaries:
            summary_df = pd.DataFrame(all_summaries)
            summary_path = self.organized_path / "data_summary.json"
            summary_json = summary_df.to_dict('records')
            with open(summary_path, 'w') as f:
                json.dump(summary_json, f, indent=2, default=str)
            logger.info(f"Saved data summary to {summary_path}")
            
            # Print summary
            print("\n" + "="*80)
            print("DATA ORGANIZATION SUMMARY")
            print("="*80)
            print(f"Total tickers processed: {len(all_summaries)}")
            print(f"Data period: {summary_df['data_start'].min()} to {summary_df['data_end'].max()}")
            print(f"Average data points per ticker: {summary_df['total_periods'].mean():.0f}")
            print(f"Total market data points: {summary_df['total_periods'].sum():,}")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("DATA ORGANIZATION SUMMARY")
            print("="*80)
            print("No data was processed successfully!")
            print("Check that data files exist in the correct locations:")
            print(f"- OHLCV data: {self.base_path}/processed/")
            print(f"- News data: {self.base_path}/news/")
            print(f"- Sentiment data: {self.base_path}/sentiment/finviz_json/")
            print("="*80)

def main():
    """Main function to organize all data"""
    # Define all tickers
    tickers = [
        'AAPL', 'MSFT', 'TSLA', 'GOOG', 'META', 'NVDA',
        'RBLX', 'AMC', 'GME', 'CROX',
        'ANF', 'URBN', 'AEO', 'LEVI', 'GPS', 'ASO', 'LULU', 'NKE',
        'JNJ', 'PFE', 'UNH', 'MRK', 'LLY', 'ABT'
    ]
    
    # Initialize organizer
    organizer = DataOrganizer()
    
    # Organize all data
    organizer.organize_all_data(tickers)

if __name__ == "__main__":
    main() 