"""
Data Preprocessor for Hybrid LSTM Model
Combines all ticker data with ticker encoding for universal model
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridDataPreprocessor:
    def __init__(self, data_dir: str = "../../data/organized/combined_data"):
        self.data_dir = Path(data_dir)
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.ticker_encoder = LabelEncoder()
        
    def load_all_ticker_data(self):
        """Load all ticker JSON files and combine them"""
        logger.info("Loading all ticker data...")
        
        # Check if combined data already exists
        combined_file = self.data_dir.parent / "combined_all_tickers.json"
        if combined_file.exists():
            logger.info("Loading existing combined data...")
            try:
                with open(combined_file, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                tickers = df['ticker'].unique().tolist()
                logger.info(f"Loaded combined data: {df.shape} records, {len(tickers)} tickers")
                return df, tickers
            except Exception as e:
                logger.error(f"Error loading combined data: {e}")
        
        all_data = []
        tickers = []
        
        # Get all organized JSON files
        json_files = list(self.data_dir.glob("*_organized.json"))
        logger.info(f"Found {len(json_files)} ticker files")
        
        for json_file in json_files:
            ticker = json_file.stem.replace('_organized', '')
            logger.info(f"Loading {ticker}...")
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if data:
                    df = pd.DataFrame(data)
                    df['ticker'] = ticker  # Add ticker identifier
                    all_data.append(df)
                    tickers.append(ticker)
                    logger.info(f"  Loaded {len(df)} records for {ticker}")
                    
            except Exception as e:
                logger.error(f"Error loading {ticker}: {e}")
        
        if not all_data:
            raise ValueError("No data loaded!")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined data shape: {combined_df.shape}")
        
        # Save combined data to JSON
        logger.info("Saving combined data to JSON...")
        combined_file = self.data_dir.parent / "combined_all_tickers.json"
        
        # Clean data for JSON serialization
        df_clean = combined_df.replace([np.inf, -np.inf], np.nan)
        json_data = df_clean.reset_index().to_dict('records')
        
        # Convert NaN to None for JSON
        for record in json_data:
            for key, value in record.items():
                if isinstance(value, (list, np.ndarray)):
                    continue
                elif pd.isna(value):
                    record[key] = None
        
        with open(combined_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        logger.info(f"Saved combined data to {combined_file}")
        
        return combined_df, tickers
    
    def encode_tickers(self, df, tickers):
        """Create one-hot encoded ticker features"""
        logger.info("Encoding tickers...")
        
        # Fit and transform ticker labels
        ticker_encoded = self.ticker_encoder.fit_transform(df['ticker'])
        
        # Create one-hot encoding
        ticker_onehot = pd.get_dummies(df['ticker'], prefix='ticker')
        
        # Add to dataframe
        df = pd.concat([df, ticker_onehot], axis=1)
        
        logger.info(f"Added {len(ticker_onehot.columns)} ticker features")
        return df
    
    def select_features(self, df):
        """Select and organize features for LSTM"""
        logger.info("Selecting features...")
        
        # Core market features
        market_features = [
            'open', 'high', 'low', 'close', 'volume', 
            'percent_change_close', 'price_change', 'volume_change', 'volatility'
        ]
        
        # Sentiment features
        sentiment_features = [
            'news_sentiment_score', 'news_count', 'news_sentiment_ratio',
            'finviz_sentiment_score', 'finviz_sentiment_ratio', 'finviz_sentiment_intensity',
            'reddit_sentiment_score', 'reddit_sentiment_ratio', 'reddit_sentiment_intensity'
        ]
        
        # Technical features
        technical_features = [
            'price_momentum_5min', 'price_momentum_15min', 'volume_momentum_5min',
            'hour_of_day', 'day_of_week', 'is_market_open'
        ]
        
        # Ticker features (one-hot encoded)
        ticker_features = [col for col in df.columns if col.startswith('ticker_')]
        
        # Target variables
        target_features = ['target_direction_5min', 'target_5min_return']
        
        # Combine all features
        all_features = market_features + sentiment_features + technical_features + ticker_features
        
        # Check which features exist in the data
        available_features = [f for f in all_features if f in df.columns]
        available_targets = [f for f in target_features if f in df.columns]
        
        logger.info(f"Selected {len(available_features)} features")
        logger.info(f"Selected {len(available_targets)} targets")
        
        return available_features, available_targets
    
    def create_sequences(self, df, features, target, sequence_length=60):
        """Create LSTM sequences with ticker encoding"""
        logger.info(f"Creating sequences with length {sequence_length}...")
        
        X, y = [], []
        
        # Group by ticker to maintain ticker context
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('index')  # Ensure chronological order
            
            # Create sequences for this ticker
            for i in range(sequence_length, len(ticker_data)):
                # Get sequence of features
                sequence = ticker_data[features].iloc[i-sequence_length:i].values
                
                # Get target
                target_value = ticker_data[target].iloc[i]
                
                X.append(sequence)
                y.append(target_value)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y
    
    def normalize_features(self, X):
        """Normalize features for LSTM"""
        logger.info("Normalizing features...")
        
        # Reshape for scaling (samples * timesteps, features)
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        # Fit and transform
        X_scaled = self.scaler.fit_transform(X_reshaped)
        
        # Reshape back
        X_normalized = X_scaled.reshape(original_shape)
        
        logger.info(f"Normalized features shape: {X_normalized.shape}")
        return X_normalized
    
    def split_data_chronologically(self, X, y, train_ratio=0.7, val_ratio=0.15):
        """Split data chronologically (not randomly)"""
        logger.info("Splitting data chronologically...")
        
        total_samples = len(X)
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_all_data(self, sequence_length=60):
        """Complete preprocessing pipeline"""
        logger.info("Starting complete preprocessing pipeline...")
        
        # 1. Load all data
        df, tickers = self.load_all_ticker_data()
        
        # 2. Encode tickers
        df = self.encode_tickers(df, tickers)
        
        # 3. Select features
        features, targets = self.select_features(df)
        
        # 4. Create sequences
        X, y = self.create_sequences(df, features, targets[0], sequence_length)  # Use direction target
        
        # 5. Normalize features
        X_normalized = self.normalize_features(X)
        
        # 6. Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data_chronologically(
            X_normalized, y
        )
        
        # 7. Save preprocessing info
        preprocessing_info = {
            'features': features,
            'target': targets[0],
            'sequence_length': sequence_length,
            'tickers': tickers,
            'scaler': self.scaler,
            'ticker_encoder': self.ticker_encoder
        }
        
        logger.info("Preprocessing completed successfully!")
        
        return {
            'X_train': X_train,
            'X_val': X_val, 
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'preprocessing_info': preprocessing_info
        }

def main():
    """Test the preprocessor"""
    preprocessor = HybridDataPreprocessor()
    data = preprocessor.preprocess_all_data(sequence_length=60)
    
    print(f"Training data shape: {data['X_train'].shape}")
    print(f"Number of features: {len(data['preprocessing_info']['features'])}")
    print(f"Number of tickers: {len(data['preprocessing_info']['tickers'])}")

if __name__ == "__main__":
    main() 