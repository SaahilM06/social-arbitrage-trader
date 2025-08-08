"""
Training Script for Hybrid LSTM Model
Trains one model on all ticker data with ticker encoding
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Import our custom modules
from data_preprocessor import HybridDataPreprocessor
from lstm_model import HybridLSTMModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridModelTrainer:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.models_dir / "ticker_models").mkdir(exist_ok=True)
        (self.models_dir / "universal_models").mkdir(exist_ok=True)
        (self.models_dir / "results").mkdir(exist_ok=True)
        
    def train_hybrid_model(self, sequence_length=60, batch_size=32, epochs=100):
        """Train the hybrid LSTM model on all ticker data"""
        logger.info("Starting hybrid model training...")
        
        # 1. Preprocess data
        logger.info("Step 1: Preprocessing data...")
        preprocessor = HybridDataPreprocessor()
        data = preprocessor.preprocess_all_data(sequence_length=sequence_length)
        
        # 2. Create model
        logger.info("Step 2: Creating LSTM model...")
        num_features = len(data['preprocessing_info']['features'])
        num_tickers = len(data['preprocessing_info']['tickers'])
        
        model = HybridLSTMModel(
            sequence_length=sequence_length,
            num_features=num_features,
            num_tickers=num_tickers
        )
        
        # Build the model
        lstm_model = model.build_model()
        
        # 3. Train model
        logger.info("Step 3: Training model...")
        history = model.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            batch_size=batch_size,
            epochs=epochs
        )
        
        # 4. Evaluate model
        logger.info("Step 4: Evaluating model...")
        results = model.evaluate(data['X_test'], data['y_test'])
        
        # 5. Save model and results
        logger.info("Step 5: Saving model and results...")
        self.save_training_results(model, data, results, history)
        
        logger.info("Training completed successfully!")
        
        return model, data, results
    
    def save_training_results(self, model, data, results, history):
        """Save the trained model and results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.models_dir / "universal_models" / f"hybrid_lstm_{timestamp}.h5"
        model.save_model(str(model_path))
        
        # Save preprocessing info
        preprocess_path = self.models_dir / "results" / f"preprocessing_info_{timestamp}.pkl"
        with open(preprocess_path, 'wb') as f:
            pickle.dump(data['preprocessing_info'], f)
        
        # Save training results
        results_path = self.models_dir / "results" / f"training_results_{timestamp}.json"
        results_to_save = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']]
            },
            'model_info': {
                'sequence_length': data['preprocessing_info']['sequence_length'],
                'num_features': len(data['preprocessing_info']['features']),
                'num_tickers': len(data['preprocessing_info']['tickers']),
                'features': data['preprocessing_info']['features'],
                'tickers': data['preprocessing_info']['tickers']
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        # Save training summary
        summary_path = self.models_dir / "results" / f"training_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("HYBRID LSTM MODEL TRAINING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model File: {model_path}\n")
            f.write(f"Sequence Length: {data['preprocessing_info']['sequence_length']}\n")
            f.write(f"Number of Features: {len(data['preprocessing_info']['features'])}\n")
            f.write(f"Number of Tickers: {len(data['preprocessing_info']['tickers'])}\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1-Score: {results['f1_score']:.4f}\n\n")
            
            f.write("CONFUSION MATRIX:\n")
            f.write(str(results['confusion_matrix']))
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"Summary saved to: {summary_path}")
    
    def analyze_feature_importance(self, model, data, sample_size=1000):
        """Analyze feature importance"""
        logger.info("Analyzing feature importance...")
        
        # Sample data for feature importance
        sample_indices = np.random.choice(len(data['X_test']), 
                                        min(sample_size, len(data['X_test'])), 
                                        replace=False)
        X_sample = data['X_test'][sample_indices]
        
        # Get feature importance
        feature_importance = model.get_feature_importance(
            X_sample, 
            data['preprocessing_info']['features']
        )
        
        # Save feature importance
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        importance_path = self.models_dir / "results" / f"feature_importance_{timestamp}.json"
        
        importance_data = {
            'feature_importance': feature_importance,
            'analysis_date': datetime.now().isoformat()
        }
        
        with open(importance_path, 'w') as f:
            json.dump(importance_data, f, indent=2)
        
        logger.info(f"Feature importance saved to: {importance_path}")
        
        # Print top features
        logger.info("Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            logger.info(f"  {i+1}. {feature}: {importance:.4f}")
        
        return feature_importance

def main():
    """Main training function"""
    logger.info("Starting Hybrid LSTM Model Training Pipeline")
    
    # Create trainer
    trainer = HybridModelTrainer()
    
    # Train model
    model, data, results = trainer.train_hybrid_model(
        sequence_length=60,  # 5 hours of 5-minute data
        batch_size=32,
        epochs=100
    )
    
    # Analyze feature importance
    feature_importance = trainer.analyze_feature_importance(model, data)
    
    # Print final results
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETED!")
    logger.info("="*50)
    logger.info(f"Final Test Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Final Test F1-Score: {results['f1_score']:.4f}")
    logger.info(f"Model trained on {len(data['preprocessing_info']['tickers'])} tickers")
    logger.info(f"Model uses {len(data['preprocessing_info']['features'])} features")
    logger.info("="*50)

if __name__ == "__main__":
    main() 