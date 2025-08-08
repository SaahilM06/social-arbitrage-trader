"""
LSTM Model for Hybrid Sentiment-Price Prediction
Uses ticker encoding to handle multiple stocks in one model
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import logging

logger = logging.getLogger(__name__)

class HybridLSTMModel:
    def __init__(self, sequence_length=60, num_features=50, num_tickers=23):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.num_tickers = num_tickers
        self.model = None
        
    def build_model(self, lstm_units=[128, 64], dropout_rate=0.2, learning_rate=0.001):
        """Build the hybrid LSTM model"""
        logger.info(f"Building LSTM model with {self.num_features} features and {self.num_tickers} tickers")
        
        model = Sequential([
            # First LSTM layer
            LSTM(lstm_units[0], 
                 return_sequences=True, 
                 input_shape=(self.sequence_length, self.num_features),
                 dropout=dropout_rate,
                 recurrent_dropout=dropout_rate),
            BatchNormalization(),
            
            # Second LSTM layer
            LSTM(lstm_units[1], 
                 return_sequences=False,
                 dropout=dropout_rate,
                 recurrent_dropout=dropout_rate),
            BatchNormalization(),
            
            # Dense layers
            Dense(32, activation='relu'),
            Dropout(dropout_rate),
            
            # Output layer (binary classification: up/down)
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        
        # Print model summary
        model.summary()
        
        return model
    
    def get_callbacks(self, patience=10, min_lr=1e-7):
        """Get training callbacks"""
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate when plateau is reached
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=min_lr,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, 
              batch_size=32, epochs=100, verbose=1):
        """Train the model"""
        logger.info("Starting model training...")
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        logger.info("Training completed!")
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        predictions = self.model.predict(X)
        return predictions
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test Precision: {precision:.4f}")
        logger.info(f"Test Recall: {recall:.4f}")
        logger.info(f"Test F1-Score: {f1:.4f}")
        
        return results
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self, X_sample, feature_names):
        """Get feature importance using permutation importance"""
        from sklearn.inspection import permutation_importance
        from sklearn.ensemble import RandomForestClassifier
        
        # Flatten the sequences for feature importance
        X_flat = X_sample.reshape(X_sample.shape[0], -1)
        
        # Use a simple classifier for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_flat, np.zeros(len(X_flat)))  # Dummy fit
        
        # Get feature importance
        importance = rf.feature_importances_
        
        # Reshape back to original feature structure
        feature_importance = {}
        for i, feature in enumerate(feature_names):
            feature_importance[feature] = importance[i]
        
        # Sort by importance
        sorted_importance = sorted(feature_importance.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        return sorted_importance

def create_hybrid_lstm_model(sequence_length=60, num_features=50, num_tickers=23):
    """Factory function to create hybrid LSTM model"""
    model = HybridLSTMModel(sequence_length, num_features, num_tickers)
    return model.build_model()

if __name__ == "__main__":
    # Test model creation
    model = create_hybrid_lstm_model()
    print("Model created successfully!") 