"""
Consumption Forecasting Model

This module implements specialized models for energy consumption forecasting
using various deep learning architectures optimized for time series data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError as MAE

logger = logging.getLogger(__name__)


@dataclass
class ConsumptionModelConfig:
    """Configuration for consumption forecasting model"""
    model_type: str = "lstm"  # lstm, gru, transformer, cnn_lstm, attention_lstm
    sequence_length: int = 24
    forecast_horizon: int = 1
    lstm_units: List[int] = None
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    loss_function: str = "mse"  # mse, mae, huber
    optimizer: str = "adam"  # adam, rmsprop
    use_attention: bool = False
    use_residual_connections: bool = True


class ConsumptionForecastingModel:
    """
    Advanced consumption forecasting model for smart meter data
    
    Supports multiple architectures:
    - LSTM with attention mechanism
    - CNN-LSTM hybrid
    - Transformer-based models
    - Ensemble methods
    """
    
    def __init__(self, config: ConsumptionModelConfig):
        self.config = config
        self.model = None
        self.history = None
        self.scaler = None
        self.feature_columns = []
        self.is_trained = False
        
        # Set default LSTM units if not provided
        if self.config.lstm_units is None:
            self.config.lstm_units = [64, 32]
    
    def create_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create LSTM model for consumption forecasting"""
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # LSTM layers
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1
            x = layers.LSTM(
                units, 
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            )(x)
            
            # Residual connection
            if self.config.use_residual_connections and i > 0:
                # Simple residual connection for LSTM
                if x.shape[-1] == inputs.shape[-1]:
                    x = layers.Add()([x, inputs])
        
        # Attention mechanism
        if self.config.use_attention:
            x = self._add_attention_layer(x, input_shape)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(self.config.forecast_horizon)(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def create_gru_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create GRU model for consumption forecasting"""
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # GRU layers
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1
            x = layers.GRU(
                units,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            )(x)
        
        # Attention mechanism
        if self.config.use_attention:
            x = self._add_attention_layer(x, input_shape)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(self.config.forecast_horizon)(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def create_cnn_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create CNN-LSTM hybrid model"""
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # CNN layers for feature extraction
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        # LSTM layers
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            )(x)
        
        # Attention mechanism
        if self.config.use_attention:
            x = self._add_attention_layer(x, input_shape)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(self.config.forecast_horizon)(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def create_transformer_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create Transformer-based model for consumption forecasting"""
        inputs = layers.Input(shape=input_shape)
        
        # Positional encoding
        x = self._add_positional_encoding(inputs)
        
        # Multi-head attention layers
        for _ in range(2):  # 2 transformer blocks
            # Self-attention
            attention_output = layers.MultiHeadAttention(
                num_heads=8,
                key_dim=64,
                dropout=self.config.dropout_rate
            )(x, x)
            
            # Add & Norm
            x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
            
            # Feed forward
            ffn = layers.Dense(128, activation='relu')(x)
            ffn = layers.Dropout(self.config.dropout_rate)(ffn)
            ffn = layers.Dense(input_shape[-1])(ffn)
            
            # Add & Norm
            x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(self.config.forecast_horizon)(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def create_attention_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create LSTM model with attention mechanism"""
        inputs = layers.Input(shape=input_shape)
        
        # LSTM layers
        lstm_outputs = []
        x = inputs
        
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = True  # Always return sequences for attention
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            )(x)
            lstm_outputs.append(x)
        
        # Attention mechanism
        attention_output = self._add_attention_layer(x, input_shape)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(attention_output)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(self.config.forecast_horizon)(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def _add_attention_layer(self, x: tf.Tensor, input_shape: Tuple[int, int]) -> tf.Tensor:
        """Add attention mechanism to the model"""
        # Self-attention
        attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dropout=self.config.dropout_rate
        )(x, x)
        
        # Add & Norm
        attention = layers.LayerNormalization(epsilon=1e-6)(x + attention)
        
        # Global average pooling
        attention_output = layers.GlobalAveragePooling1D()(attention)
        
        return attention_output
    
    def _add_positional_encoding(self, inputs: tf.Tensor) -> tf.Tensor:
        """Add positional encoding for transformer models"""
        seq_len = tf.shape(inputs)[1]
        d_model = inputs.shape[-1]
        
        # Create positional encoding
        pos_encoding = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        
        pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
        
        return inputs + pos_encoding
    
    def compile_model(self, model: keras.Model) -> keras.Model:
        """Compile the model with specified optimizer and loss"""
        # Select optimizer
        if self.config.optimizer == "adam":
            optimizer = Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == "rmsprop":
            optimizer = RMSprop(learning_rate=self.config.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        # Select loss function
        if self.config.loss_function == "mse":
            loss = MeanSquaredError()
        elif self.config.loss_function == "mae":
            loss = MeanAbsoluteError()
        elif self.config.loss_function == "huber":
            loss = Huber()
        else:
            raise ValueError(f"Unsupported loss function: {self.config.loss_function}")
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[RootMeanSquaredError(), MAE()]
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train the consumption forecasting model"""
        logger.info(f"Training consumption forecasting model: {self.config.model_type}")
        
        # Create model
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        if self.config.model_type == "lstm":
            model = self.create_lstm_model(input_shape)
        elif self.config.model_type == "gru":
            model = self.create_gru_model(input_shape)
        elif self.config.model_type == "cnn_lstm":
            model = self.create_cnn_lstm_model(input_shape)
        elif self.config.model_type == "transformer":
            model = self.create_transformer_model(input_shape)
        elif self.config.model_type == "attention_lstm":
            model = self.create_attention_lstm_model(input_shape)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        # Compile model
        model = self.compile_model(model)
        
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
            ),
            callbacks.ModelCheckpoint(
                filepath='best_consumption_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Store model and history
        self.model = model
        self.history = history.history
        self.is_trained = True
        
        logger.info("Consumption forecasting model training completed")
        
        return {
            "model": model,
            "history": history.history,
            "config": self.config
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        # Calculate R²
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "r2": float(r2)
        }
        
        logger.info(f"Model evaluation completed. RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return metrics
    
    def predict_with_confidence(self, X: np.ndarray, n_samples: int = 100) -> Dict[str, np.ndarray]:
        """Make predictions with confidence intervals using Monte Carlo dropout"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Enable dropout during inference
        self.model.trainable = True
        
        predictions = []
        for _ in range(n_samples):
            pred = self.model(X, training=True)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Calculate confidence intervals
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred
        
        return {
            "predictions": mean_pred,
            "std": std_pred,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "confidence_95": upper_bound - lower_bound
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if not self.is_trained:
            return "Model not trained"
        
        return self.model.summary()
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history"""
        if not self.is_trained:
            return {}
        
        return self.history
