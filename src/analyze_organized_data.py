"""
Analyze Organized Data Sources
Count and analyze different data sources in the organized data
"""

import json
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_organized_data(data_dir: str = "../data/organized/combined_data"):
    """
    Analyze organized data to count sources and assess data quality
    """
    
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory {data_dir} does not exist")
        return
    
    # Get all organized JSON files
    json_files = list(data_path.glob("*_organized.json"))
    logger.info(f"Found {len(json_files)} organized data files")
    
    total_records = 0
    source_counts = {
        'market_data': 0,
        'news_data': 0,
        'finviz_sentiment': 0,
        'reddit_sentiment': 0
    }
    
    ticker_summaries = []
    
    for json_file in json_files:
        ticker = json_file.stem.replace('_organized', '')
        logger.info(f"Analyzing {ticker}...")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if not data:
                logger.warning(f"No data found in {json_file}")
                continue
            
            df = pd.DataFrame(data)
            total_records += len(df)
            
            # Count records with data from each source
            ticker_summary = {
                'ticker': ticker,
                'total_records': len(df),
                'market_data_records': len(df[df['open'].notna()]),
                'news_data_records': len(df[df['news_count'] > 0]) if 'news_count' in df.columns else 0,
                'finviz_sentiment_records': len(df[df['total_finviz_sentiment_items'] > 0]) if 'total_finviz_sentiment_items' in df.columns else 0,
                'reddit_sentiment_records': len(df[df['total_reddit_sentiment_items'] > 0]) if 'total_reddit_sentiment_items' in df.columns else 0,
                'date_range_start': df['timestamp'].min() if 'timestamp' in df.columns else None,
                'date_range_end': df['timestamp'].max() if 'timestamp' in df.columns else None
            }
            
            # Update global source counts
            source_counts['market_data'] += ticker_summary['market_data_records']
            source_counts['news_data'] += ticker_summary['news_data_records']
            source_counts['finviz_sentiment'] += ticker_summary['finviz_sentiment_records']
            source_counts['reddit_sentiment'] += ticker_summary['reddit_sentiment_records']
            
            ticker_summaries.append(ticker_summary)
            
            logger.info(f"  {ticker}: {len(df)} records")
            logger.info(f"    Market data: {ticker_summary['market_data_records']}")
            logger.info(f"    News data: {ticker_summary['news_data_records']}")
            logger.info(f"    Finviz sentiment: {ticker_summary['finviz_sentiment_records']}")
            logger.info(f"    Reddit sentiment: {ticker_summary['reddit_sentiment_records']}")
            
        except Exception as e:
            logger.error(f"Error analyzing {json_file}: {e}")
    
    # Print overall summary
    logger.info("\n" + "="*80)
    logger.info("OVERALL DATA SOURCE ANALYSIS")
    logger.info("="*80)
    logger.info(f"Total records across all tickers: {total_records:,}")
    logger.info(f"Total tickers analyzed: {len(ticker_summaries)}")
    
    logger.info("\nSOURCE BREAKDOWN:")
    logger.info(f"Market data records: {source_counts['market_data']:,}")
    logger.info(f"News data records: {source_counts['news_data']:,}")
    logger.info(f"Finviz sentiment records: {source_counts['finviz_sentiment']:,}")
    logger.info(f"Reddit sentiment records: {source_counts['reddit_sentiment']:,}")
    
    # Calculate coverage percentages
    if total_records > 0:
        logger.info("\nCOVERAGE PERCENTAGES:")
        logger.info(f"Market data coverage: {source_counts['market_data']/total_records*100:.1f}%")
        logger.info(f"News data coverage: {source_counts['news_data']/total_records*100:.1f}%")
        logger.info(f"Finviz sentiment coverage: {source_counts['finviz_sentiment']/total_records*100:.1f}%")
        logger.info(f"Reddit sentiment coverage: {source_counts['reddit_sentiment']/total_records*100:.1f}%")
    
    # Analyze tickers with best data coverage
    if ticker_summaries:
        summary_df = pd.DataFrame(ticker_summaries)
        
        # Calculate total sentiment coverage per ticker
        summary_df['total_sentiment_records'] = (
            summary_df['finviz_sentiment_records'] + 
            summary_df['reddit_sentiment_records']
        )
        
        # Sort by total sentiment records
        best_coverage = summary_df.nlargest(10, 'total_sentiment_records')
        
        logger.info("\nTOP 10 TICKERS BY SENTIMENT DATA COVERAGE:")
        for _, row in best_coverage.iterrows():
            logger.info(f"  {row['ticker']}: {row['total_sentiment_records']} sentiment records "
                       f"(Finviz: {row['finviz_sentiment_records']}, Reddit: {row['reddit_sentiment_records']})")
        
        # Count tickers with each source
        tickers_with_news = len(summary_df[summary_df['news_data_records'] > 0])
        tickers_with_finviz = len(summary_df[summary_df['finviz_sentiment_records'] > 0])
        tickers_with_reddit = len(summary_df[summary_df['reddit_sentiment_records'] > 0])
        
        logger.info(f"\nTICKER COVERAGE:")
        logger.info(f"Tickers with news data: {tickers_with_news}/{len(summary_df)}")
        logger.info(f"Tickers with Finviz sentiment: {tickers_with_finviz}/{len(summary_df)}")
        logger.info(f"Tickers with Reddit sentiment: {tickers_with_reddit}/{len(summary_df)}")
    
    return {
        'total_records': total_records,
        'source_counts': source_counts,
        'ticker_summaries': ticker_summaries
    }

def main():
    """Main function"""
    analyze_organized_data()

if __name__ == "__main__":
    main() 