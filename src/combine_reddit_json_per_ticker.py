"""
Combine Reddit JSON Sentiment Data Per Ticker
Merges all monthly JSON files for each ticker into separate combined files
"""

import json
import os
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def combine_reddit_json_per_ticker(input_dir: str = "../data/sentiment/reddit_json", 
                                 output_dir: str = "../data/sentiment/reddit_combined"):
    """
    Combine Reddit JSON sentiment data per ticker
    
    Args:
        input_dir: Directory containing ticker subdirectories with monthly JSON files
        output_dir: Directory to save combined JSON files (one per ticker)
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all ticker directories
    ticker_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(ticker_dirs)} ticker directories")
    
    total_records = 0
    
    for ticker_dir in ticker_dirs:
        ticker = ticker_dir.name
        logger.info(f"Processing {ticker}...")
        
        # Get all JSON files in the ticker directory
        json_files = list(ticker_dir.glob("*.json"))
        logger.info(f"  Found {len(json_files)} JSON files for {ticker}")
        
        if not json_files:
            logger.warning(f"  No JSON files found for {ticker}")
            continue
        
        ticker_data = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    # Add source file info to each record
                    for record in data:
                        if isinstance(record, dict):
                            record['source_file'] = str(json_file.name)  # Just the filename
                            ticker_data.append(record)
                else:
                    logger.warning(f"Unexpected data format in {json_file}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Error reading {json_file}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error reading {json_file}: {e}")
        
        if ticker_data:
            # Sort by timestamp
            ticker_data.sort(key=lambda x: x.get('timestamp', ''))
            
            # Save combined data for this ticker
            output_file = output_path / f"{ticker}_combined.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(ticker_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"  Saved {len(ticker_data)} records for {ticker} to {output_file}")
            total_records += len(ticker_data)
            
            # Print date range for this ticker
            timestamps = [record.get('timestamp') for record in ticker_data if record.get('timestamp')]
            if timestamps:
                timestamps.sort()
                logger.info(f"  Date range: {timestamps[0]} to {timestamps[-1]}")
        else:
            logger.warning(f"  No valid data found for {ticker}")
    
    # Print final summary
    logger.info("=== FINAL SUMMARY ===")
    logger.info(f"Total records across all tickers: {total_records}")
    logger.info(f"Total tickers processed: {len(ticker_dirs)}")
    logger.info(f"Combined files saved to: {output_dir}")
    
    # List all created files
    combined_files = list(output_path.glob("*_combined.json"))
    logger.info(f"Created {len(combined_files)} combined files:")
    for file in combined_files:
        file_size = file.stat().st_size / 1024  # Size in KB
        logger.info(f"  {file.name} ({file_size:.1f} KB)")

def main():
    """Main function"""
    combine_reddit_json_per_ticker()

if __name__ == "__main__":
    main() 