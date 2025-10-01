"""
Anomaly Detection Model

This module implements advanced anomaly detection models for smart meter data
using autoencoders, isolation forests, and other unsupervised learning methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


@dataclass
class AnomalyModelConfig:
    """Configuration for anomaly detection model"""
    model_type: str = "autoencoder"  # autoencoder, isolation_forest, one_class_svm, lstm_autoencoder
    threshold_method: str = "percentile"  # percentile, std, iqr
    threshold_value: float = 95.0  # For percentile method
    contamination: float = 0.1  # For isolation forest
    sequence_length: int = 24
    encoding_dim: int = 8
    hidden_layers: List[int] = None
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    use_attention: bool = False


class AnomalyDetectionModel:
    """
    Advanced anomaly detection model for smart meter data
    
    Supports multiple approaches:
    - Autoencoder-based reconstruction error
    - Isolation Forest for outlier detection
    - One-Class SVM for novelty detection
    - LSTM-based autoencoder for time series
    """
    
    def __init__(self, config: AnomalyModelConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.threshold = None
        self.is_trained = False
        
        # Set default hidden layers if not provided
        if self.config.hidden_layers is None:
            self.config.hidden_layers = [64, 32, 16]
    
    def create_autoencoder(self, input_shape: Tuple[int, int]) -> Tuple[keras.Model, keras.Model, keras.Model]:
        """Create autoencoder model for anomaly detection"""
        input_dim = input_shape[0] * input_shape[1] if len(input_shape) > 1 else input_shape[0]
        
        # Encoder
        encoder_input = layers.Input(shape=input_shape, name='encoder_input')
        
        if len(input_shape) > 1:
            # For time series data, flatten first
            x = layers.Flatten()(encoder_input)
        else:
            x = encoder_input
        
        # Hidden layers
        for i, units in enumerate(self.config.hidden_layers):
            x = layers.Dense(units, activation='relu', name=f'encoder_dense_{i}')(x)
            x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Latent space
        encoded = layers.Dense(self.config.encoding_dim, activation='relu', name='encoded')(x)
        
        # Decoder
        decoder_input = layers.Input(shape=(self.config.encoding_dim,), name='decoder_input')
        x = decoder_input
        
        # Reverse hidden layers
        for i, units in enumerate(reversed(self.config.hidden_layers)):
            x = layers.Dense(units, activation='relu', name=f'decoder_dense_{i}')(x)
            x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Output layer
        if len(input_shape) > 1:
            decoded = layers.Dense(input_dim, activation='sigmoid', name='decoded')(x)
            decoded = layers.Reshape(input_shape)(decoded)
        else:
            decoded = layers.Dense(input_dim, activation='sigmoid', name='decoded')(x)
        
        # Create models
        encoder = keras.Model(encoder_input, encoded, name='encoder')
        decoder = keras.Model(decoder_input, decoded, name='decoder')
        
        # Autoencoder
        autoencoder_input = layers.Input(shape=input_shape, name='autoencoder_input')
        encoded_output = encoder(autoencoder_input)
        decoded_output = decoder(encoded_output)
        autoencoder = keras.Model(autoencoder_input, decoded_output, name='autoencoder')
        
        return autoencoder, encoder, decoder
    
    def create_lstm_autoencoder(self, input_shape: Tuple[int, int]) -> Tuple[keras.Model, keras.Model, keras.Model]:
        """Create LSTM-based autoencoder for time series anomaly detection"""
        # Encoder
        encoder_input = layers.Input(shape=input_shape, name='encoder_input')
        
        # LSTM encoder layers
        x = layers.LSTM(64, return_sequences=True, dropout=self.config.dropout_rate)(encoder_input)
        x = layers.LSTM(32, return_sequences=True, dropout=self.config.dropout_rate)(x)
        x = layers.LSTM(16, return_sequences=False, dropout=self.config.dropout_rate)(x)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        encoded = layers.Dense(self.config.encoding_dim, activation='relu', name='encoded')(x)
        
        # Decoder
        decoder_input = layers.Input(shape=(self.config.encoding_dim,), name='decoder_input')
        x = decoder_input
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.RepeatVector(input_shape[0])(x)  # Repeat for sequence length
        
        # LSTM decoder layers
        x = layers.LSTM(16, return_sequences=True, dropout=self.config.dropout_rate)(x)
        x = layers.LSTM(32, return_sequences=True, dropout=self.config.dropout_rate)(x)
        decoded = layers.LSTM(input_shape[1], return_sequences=True, dropout=self.config.dropout_rate)(x)
        
        # Create models
        encoder = keras.Model(encoder_input, encoded, name='encoder')
        decoder = keras.Model(decoder_input, decoded, name='decoder')
        
        # Autoencoder
        autoencoder_input = layers.Input(shape=input_shape, name='autoencoder_input')
        encoded_output = encoder(autoencoder_input)
        decoded_output = decoder(encoded_output)
        autoencoder = keras.Model(autoencoder_input, decoded_output, name='autoencoder')
        
        return autoencoder, encoder, decoder
    
    def create_attention_autoencoder(self, input_shape: Tuple[int, int]) -> Tuple[keras.Model, keras.Model, keras.Model]:
        """Create autoencoder with attention mechanism"""
        # Encoder
        encoder_input = layers.Input(shape=input_shape, name='encoder_input')
        
        # LSTM layers with attention
        x = layers.LSTM(64, return_sequences=True, dropout=self.config.dropout_rate)(encoder_input)
        x = layers.LSTM(32, return_sequences=True, dropout=self.config.dropout_rate)(x)
        
        # Attention mechanism
        attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=16,
            dropout=self.config.dropout_rate
        )(x, x)
        
        # Add & Norm
        attention = layers.LayerNormalization(epsilon=1e-6)(x + attention)
        
        # Global average pooling
        attention_output = layers.GlobalAveragePooling1D()(attention)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(attention_output)
        x = layers.Dropout(self.config.dropout_rate)(x)
        encoded = layers.Dense(self.config.encoding_dim, activation='relu', name='encoded')(x)
        
        # Decoder
        decoder_input = layers.Input(shape=(self.config.encoding_dim,), name='decoder_input')
        x = decoder_input
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.RepeatVector(input_shape[0])(x)
        
        # LSTM decoder
        x = layers.LSTM(32, return_sequences=True, dropout=self.config.dropout_rate)(x)
        decoded = layers.LSTM(input_shape[1], return_sequences=True, dropout=self.config.dropout_rate)(x)
        
        # Create models
        encoder = keras.Model(encoder_input, encoded, name='encoder')
        decoder = keras.Model(decoder_input, decoded, name='decoder')
        
        # Autoencoder
        autoencoder_input = layers.Input(shape=input_shape, name='autoencoder_input')
        encoded_output = encoder(autoencoder_input)
        decoded_output = decoder(encoded_output)
        autoencoder = keras.Model(autoencoder_input, decoded_output, name='autoencoder')
        
        return autoencoder, encoder, decoder
    
    def train_autoencoder(self, X_train: np.ndarray, X_val: np.ndarray) -> Dict[str, Any]:
        """Train autoencoder model"""
        logger.info(f"Training autoencoder model: {self.config.model_type}")
        
        # Create model
        input_shape = X_train.shape[1:]
        
        if self.config.model_type == "autoencoder":
            autoencoder, encoder, decoder = self.create_autoencoder(input_shape)
        elif self.config.model_type == "lstm_autoencoder":
            autoencoder, encoder, decoder = self.create_lstm_autoencoder(input_shape)
        elif self.config.model_type == "attention_autoencoder":
            autoencoder, encoder, decoder = self.create_attention_autoencoder(input_shape)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        # Compile model
        autoencoder.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss=MeanSquaredError(),
            metrics=['mae']
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        history = autoencoder.fit(
            X_train, X_train,  # Autoencoder learns to reconstruct input
            validation_data=(X_val, X_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Calculate reconstruction error on training data
        train_pred = autoencoder.predict(X_train, verbose=0)
        train_error = np.mean(np.square(X_train - train_pred), axis=1)
        
        # Calculate threshold
        self.threshold = self._calculate_threshold(train_error)
        
        # Store model
        self.model = autoencoder
        self.is_trained = True
        
        logger.info(f"Autoencoder training completed. Threshold: {self.threshold:.4f}")
        
        return {
            "model": autoencoder,
            "encoder": encoder,
            "decoder": decoder,
            "history": history.history,
            "threshold": self.threshold
        }
    
    def train_isolation_forest(self, X_train: np.ndarray) -> Dict[str, Any]:
        """Train Isolation Forest model"""
        logger.info("Training Isolation Forest model")
        
        # Flatten data if needed
        if len(X_train.shape) > 2:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_flat = X_train
        
        # Scale data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        
        # Train model
        model = IsolationForest(
            contamination=self.config.contamination,
            random_state=42,
            n_estimators=100
        )
        model.fit(X_train_scaled)
        
        # Store model
        self.model = model
        self.is_trained = True
        
        logger.info("Isolation Forest training completed")
        
        return {
            "model": model,
            "scaler": self.scaler
        }
    
    def train_one_class_svm(self, X_train: np.ndarray) -> Dict[str, Any]:
        """Train One-Class SVM model"""
        logger.info("Training One-Class SVM model")
        
        # Flatten data if needed
        if len(X_train.shape) > 2:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_flat = X_train
        
        # Scale data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        
        # Train model
        model = OneClassSVM(
            nu=self.config.contamination,
            kernel='rbf',
            gamma='scale'
        )
        model.fit(X_train_scaled)
        
        # Store model
        self.model = model
        self.is_trained = True
        
        logger.info("One-Class SVM training completed")
        
        return {
            "model": model,
            "scaler": self.scaler
        }
    
    def train(self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the anomaly detection model"""
        if self.config.model_type in ["autoencoder", "lstm_autoencoder", "attention_autoencoder"]:
            if X_val is None:
                # Split training data for validation
                split_idx = int(0.8 * len(X_train))
                X_val = X_train[split_idx:]
                X_train = X_train[:split_idx]
            return self.train_autoencoder(X_train, X_val)
        elif self.config.model_type == "isolation_forest":
            return self.train_isolation_forest(X_train)
        elif self.config.model_type == "one_class_svm":
            return self.train_one_class_svm(X_train)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.config.model_type in ["autoencoder", "lstm_autoencoder", "attention_autoencoder"]:
            return self._predict_autoencoder(X)
        elif self.config.model_type == "isolation_forest":
            return self._predict_isolation_forest(X)
        elif self.config.model_type == "one_class_svm":
            return self._predict_one_class_svm(X)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _predict_autoencoder(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies using autoencoder"""
        # Reconstruct data
        X_pred = self.model.predict(X, verbose=0)
        
        # Calculate reconstruction error
        reconstruction_error = np.mean(np.square(X - X_pred), axis=1)
        
        # Detect anomalies
        anomalies = reconstruction_error > self.threshold
        
        return anomalies.astype(int)
    
    def _predict_isolation_forest(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies using Isolation Forest"""
        # Flatten data if needed
        if len(X.shape) > 2:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
        
        # Scale data
        X_scaled = self.scaler.transform(X_flat)
        
        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = self.model.predict(X_scaled)
        
        # Convert to binary (1 for anomaly, 0 for normal)
        anomalies = (predictions == -1).astype(int)
        
        return anomalies
    
    def _predict_one_class_svm(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies using One-Class SVM"""
        # Flatten data if needed
        if len(X.shape) > 2:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
        
        # Scale data
        X_scaled = self.scaler.transform(X_flat)
        
        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = self.model.predict(X_scaled)
        
        # Convert to binary (1 for anomaly, 0 for normal)
        anomalies = (predictions == -1).astype(int)
        
        return anomalies
    
    def predict_with_scores(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict anomalies with anomaly scores"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.config.model_type in ["autoencoder", "lstm_autoencoder", "attention_autoencoder"]:
            # Reconstruct data
            X_pred = self.model.predict(X, verbose=0)
            
            # Calculate reconstruction error
            reconstruction_error = np.mean(np.square(X - X_pred), axis=1)
            
            # Normalize scores to 0-1 range
            scores = reconstruction_error / (np.max(reconstruction_error) + 1e-8)
            
            # Detect anomalies
            anomalies = reconstruction_error > self.threshold
            
            return {
                "anomalies": anomalies.astype(int),
                "scores": scores,
                "reconstruction_error": reconstruction_error
            }
        
        elif self.config.model_type == "isolation_forest":
            # Flatten data if needed
            if len(X.shape) > 2:
                X_flat = X.reshape(X.shape[0], -1)
            else:
                X_flat = X
            
            # Scale data
            X_scaled = self.scaler.transform(X_flat)
            
            # Get anomaly scores
            scores = self.model.decision_function(X_scaled)
            
            # Normalize scores to 0-1 range
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
            
            # Predict anomalies
            predictions = self.model.predict(X_scaled)
            anomalies = (predictions == -1).astype(int)
            
            return {
                "anomalies": anomalies,
                "scores": scores
            }
        
        elif self.config.model_type == "one_class_svm":
            # Flatten data if needed
            if len(X.shape) > 2:
                X_flat = X.reshape(X.shape[0], -1)
            else:
                X_flat = X
            
            # Scale data
            X_scaled = self.scaler.transform(X_flat)
            
            # Get anomaly scores
            scores = self.model.decision_function(X_scaled)
            
            # Normalize scores to 0-1 range
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
            
            # Predict anomalies
            predictions = self.model.predict(X_scaled)
            anomalies = (predictions == -1).astype(int)
            
            return {
                "anomalies": anomalies,
                "scores": scores
            }
    
    def _calculate_threshold(self, reconstruction_error: np.ndarray) -> float:
        """Calculate anomaly threshold"""
        if self.config.threshold_method == "percentile":
            return np.percentile(reconstruction_error, self.config.threshold_value)
        elif self.config.threshold_method == "std":
            return np.mean(reconstruction_error) + self.config.threshold_value * np.std(reconstruction_error)
        elif self.config.threshold_method == "iqr":
            Q1 = np.percentile(reconstruction_error, 25)
            Q3 = np.percentile(reconstruction_error, 75)
            IQR = Q3 - Q1
            return Q3 + self.config.threshold_value * IQR
        else:
            raise ValueError(f"Unsupported threshold method: {self.config.threshold_method}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model on test data"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Predict anomalies
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist()
        }
        
        logger.info(f"Model evaluation completed. F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        import joblib
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "threshold": self.threshold,
            "config": self.config,
            "is_trained": self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.threshold = model_data["threshold"]
        self.is_trained = model_data["is_trained"]
        
        logger.info(f"Model loaded from {filepath}")
