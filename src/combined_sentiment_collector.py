"""
Combined Sentiment Collector for Social Arbitrage Trader
Integrates all sentiment sources to provide comprehensive coverage from April to now
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CombinedSentimentCollector:
    def __init__(self):
        self.base_path = Path("data")
        self.sentiment_sources = {
            'news': self.base_path / "news",
            'finviz': self.base_path / "sentiment" / "finviz_json",
            'twitter': self.base_path / "sentiment" / "twitter_json",
            'reddit': self.base_path / "sentiment" / "reddit_json"
        }
        
    def get_sentiment_coverage_report(self):
        """Generate a report of sentiment data coverage by source and date"""
        logger.info("Generating sentiment coverage report...")
        
        coverage_data = {}
        
        for source_name, source_path in self.sentiment_sources.items():
            if not source_path.exists():
                logger.warning(f"Source {source_name} not found at {source_path}")
                continue
                
            source_coverage = self._analyze_source_coverage(source_name, source_path)
            coverage_data[source_name] = source_coverage
        
        return coverage_data
    
    def _analyze_source_coverage(self, source_name: str, source_path: Path) -> Dict:
        """Analyze coverage for a specific sentiment source"""
        coverage = {
            'total_tickers': 0,
            'date_range': {'start': None, 'end': None},
            'total_records': 0,
            'records_by_month': {},
            'tickers_with_data': []
        }
        
        ticker_dirs = [d for d in source_path.iterdir() if d.is_dir()]
        coverage['total_tickers'] = len(ticker_dirs)
        
        all_dates = []
        all_records = []
        
        for ticker_dir in ticker_dirs:
            ticker = ticker_dir.name
            ticker_records = 0
            
            # Check monthly files
            for json_file in ticker_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list) and data:
                        ticker_records += len(data)
                        all_records.extend(data)
                        
                        # Extract dates
                        for record in data:
                            if 'timestamp' in record:
                                try:
                                    date = datetime.strptime(record['timestamp'], '%Y-%m-%d %H:%M:%S')
                                    all_dates.append(date)
                                except:
                                    pass
                            
                except Exception as e:
                    logger.error(f"Error reading {json_file}: {e}")
            
            if ticker_records > 0:
                coverage['tickers_with_data'].append({
                    'ticker': ticker,
                    'records': ticker_records
                })
        
        # Calculate date range
        if all_dates:
            coverage['date_range']['start'] = min(all_dates).strftime('%Y-%m-%d')
            coverage['date_range']['end'] = max(all_dates).strftime('%Y-%m-%d')
        
        coverage['total_records'] = len(all_records)
        
        # Group by month
        for record in all_records:
            if 'timestamp' in record:
                try:
                    date = datetime.strptime(record['timestamp'], '%Y-%m-%d %H:%M:%S')
                    month_key = date.strftime('%Y-%m')
                    coverage['records_by_month'][month_key] = coverage['records_by_month'].get(month_key, 0) + 1
                except:
                    pass
        
        return coverage
    
    def print_coverage_report(self):
        """Print a formatted coverage report"""
        coverage_data = self.get_sentiment_coverage_report()
        
        print("\n" + "="*80)
        print("SENTIMENT DATA COVERAGE REPORT")
        print("="*80)
        
        total_records = 0
        for source_name, coverage in coverage_data.items():
            print(f"\n📊 {source_name.upper()} SENTIMENT:")
            print(f"   Total Records: {coverage['total_records']:,}")
            print(f"   Tickers with Data: {len(coverage['tickers_with_data'])}/{coverage['total_tickers']}")
            
            if coverage['date_range']['start']:
                print(f"   Date Range: {coverage['date_range']['start']} to {coverage['date_range']['end']}")
            
            if coverage['records_by_month']:
                print("   Records by Month:")
                for month, count in sorted(coverage['records_by_month'].items()):
                    print(f"     {month}: {count:,} records")
            
            total_records += coverage['total_records']
        
        print(f"\n📈 TOTAL SENTIMENT RECORDS: {total_records:,}")
        
        # Check April coverage
        april_coverage = 0
        for source_name, coverage in coverage_data.items():
            april_records = coverage['records_by_month'].get('2025-04', 0)
            april_coverage += april_records
        
        print(f"📅 APRIL 2025 COVERAGE: {april_coverage:,} records")
        
        if april_coverage == 0:
            print("⚠️  WARNING: No sentiment data for April 2025!")
            print("💡 RECOMMENDATION: Use news data as primary sentiment source")
        
        print("="*80)
    
    def get_recommended_strategy(self):
        """Get recommended strategy based on data coverage"""
        coverage_data = self.get_sentiment_coverage_report()
        
        # Count records by source
        source_counts = {name: data['total_records'] for name, data in coverage_data.items()}
        
        # Check April coverage
        april_coverage = 0
        for source_name, coverage in coverage_data.items():
            april_records = coverage['records_by_month'].get('2025-04', 0)
            april_coverage += april_records
        
        recommendations = []
        
        if april_coverage == 0:
            recommendations.append("🚨 CRITICAL: No sentiment data for April-June 2025")
            recommendations.append("✅ SOLUTION: Use news data as primary sentiment source")
            recommendations.append("✅ SOLUTION: Start collecting Twitter/Reddit data now for future models")
        
        # Find best data source
        best_source = max(source_counts.items(), key=lambda x: x[1])
        recommendations.append(f"📊 BEST DATA SOURCE: {best_source[0]} ({best_source[1]:,} records)")
        
        # ML strategy recommendations
        if april_coverage > 0:
            recommendations.append("🎯 ML STRATEGY: Use all sentiment sources with proper weighting")
        else:
            recommendations.append("🎯 ML STRATEGY: Focus on news sentiment + technical indicators")
            recommendations.append("🎯 ML STRATEGY: Use Twitter/Reddit as supplementary features")
        
        return recommendations

def main():
    """Main function to run the combined sentiment analysis"""
    collector = CombinedSentimentCollector()
    collector.print_coverage_report()
    
    print("\n" + "="*80)
    print("RECOMMENDED STRATEGY")
    print("="*80)
    
    recommendations = collector.get_recommended_strategy()
    for rec in recommendations:
        print(f"   {rec}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main() 