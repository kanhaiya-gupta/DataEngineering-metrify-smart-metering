"""
Model Training Pipeline

This module implements automated model training for smart meter data,
including consumption forecasting, anomaly detection, and grid optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime
import joblib
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError as MAE

import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    learning_rate: float = 0.001
    model_type: str = "lstm"  # lstm, gru, transformer, cnn_lstm
    sequence_length: int = 24
    forecast_horizon: int = 1


class ModelTrainingPipeline:
    """
    Automated model training pipeline for smart meter data
    
    Supports multiple model types:
    - LSTM for time series forecasting
    - Autoencoder for anomaly detection
    - CNN-LSTM for complex patterns
    - Transformer for long-term dependencies
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.models = {}
        self.history = {}
        self.metrics = {}
        
        # Set up MLflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("smart_meter_ml")
    
    def create_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create LSTM model for time series forecasting"""
        model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(50),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(self.config.forecast_horizon)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss=MeanSquaredError(),
            metrics=[RootMeanSquaredError(), MAE()]
        )
        
        return model
    
    def create_gru_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create GRU model for time series forecasting"""
        model = keras.Sequential([
            layers.GRU(50, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.GRU(50, return_sequences=True),
            layers.Dropout(0.2),
            layers.GRU(50),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(self.config.forecast_horizon)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss=MeanSquaredError(),
            metrics=[RootMeanSquaredError(), MAE()]
        )
        
        return model
    
    def create_cnn_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create CNN-LSTM model for complex pattern recognition"""
        model = keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(50),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(self.config.forecast_horizon)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss=MeanSquaredError(),
            metrics=[RootMeanSquaredError(), MAE()]
        )
        
        return model
    
    def create_transformer_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create Transformer model for long-term dependencies"""
        inputs = layers.Input(shape=input_shape)
        
        # Multi-head attention
        attention = layers.MultiHeadAttention(
            num_heads=8, 
            key_dim=64, 
            dropout=0.2
        )(inputs, inputs)
        
        # Add & Norm
        attention = layers.Dropout(0.2)(attention)
        attention = layers.LayerNormalization(epsilon=1e-6)(attention + inputs)
        
        # Feed forward
        ffn = layers.Dense(128, activation='relu')(attention)
        ffn = layers.Dropout(0.2)(ffn)
        ffn = layers.Dense(input_shape[-1])(ffn)
        ffn = layers.LayerNormalization(epsilon=1e-6)(ffn + attention)
        
        # Global average pooling and output
        outputs = layers.GlobalAveragePooling1D()(ffn)
        outputs = layers.Dense(25, activation='relu')(outputs)
        outputs = layers.Dropout(0.2)(outputs)
        outputs = layers.Dense(self.config.forecast_horizon)(outputs)
        
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss=MeanSquaredError(),
            metrics=[RootMeanSquaredError(), MAE()]
        )
        
        return model
    
    def create_autoencoder_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create autoencoder model for anomaly detection"""
        # Encoder
        encoder = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=input_shape),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu')  # Latent space
        ])
        
        # Decoder
        decoder = keras.Sequential([
            layers.Dense(16, activation='relu', input_shape=(8,)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(input_shape[0], activation='sigmoid')
        ])
        
        # Autoencoder
        inputs = layers.Input(shape=input_shape)
        encoded = encoder(inputs)
        decoded = decoder(encoded)
        
        autoencoder = keras.Model(inputs, decoded)
        autoencoder.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder, encoder, decoder
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""
        # Time series split to maintain temporal order
        tscv = TimeSeriesSplit(n_splits=1, test_size=int(len(X) * self.config.test_size))
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        
        # Further split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=self.config.validation_size,
            random_state=self.config.random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_consumption_forecasting_model(self, X: np.ndarray, y: np.ndarray, 
                                          model_name: str = "consumption_forecasting") -> Dict[str, Any]:
        """Train consumption forecasting model"""
        logger.info(f"Training {model_name} model")
        
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_params({
                "model_type": self.config.model_type,
                "sequence_length": self.config.sequence_length,
                "forecast_horizon": self.config.forecast_horizon,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate
            })
            
            # Prepare data
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(X, y)
            
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
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            # Callbacks
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
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
            
            # Evaluate model
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, train_pred)
            val_metrics = self._calculate_metrics(y_val, val_pred)
            test_metrics = self._calculate_metrics(y_test, test_pred)
            
            # Log metrics
            mlflow.log_metrics({
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "train_r2": train_metrics["r2"],
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_r2": test_metrics["r2"]
            })
            
            # Save model
            model_path = f"models/{model_name}"
            model.save(model_path)
            mlflow.tensorflow.log_model(model, "model")
            
            # Store results
            self.models[model_name] = model
            self.history[model_name] = history.history
            self.metrics[model_name] = {
                "train": train_metrics,
                "validation": val_metrics,
                "test": test_metrics
            }
            
            logger.info(f"Model {model_name} trained successfully")
            logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
            logger.info(f"Test MAE: {test_metrics['mae']:.4f}")
            logger.info(f"Test R²: {test_metrics['r2']:.4f}")
            
            return {
                "model": model,
                "history": history.history,
                "metrics": {
                    "train": train_metrics,
                    "validation": val_metrics,
                    "test": test_metrics
                }
            }
    
    def train_anomaly_detection_model(self, X: np.ndarray, model_name: str = "anomaly_detection") -> Dict[str, Any]:
        """Train anomaly detection model using autoencoder"""
        logger.info(f"Training {model_name} model")
        
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_params({
                "model_type": "autoencoder",
                "sequence_length": self.config.sequence_length,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate
            })
            
            # Prepare data (no need for y in autoencoder)
            X_train, X_val, X_test, _, _, _ = self.prepare_data(X, X)
            
            # Create autoencoder model
            input_shape = (X_train.shape[1], X_train.shape[2])
            autoencoder, encoder, decoder = self.create_autoencoder_model(input_shape)
            
            # Callbacks
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
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
            
            # Evaluate model
            train_pred = autoencoder.predict(X_train)
            val_pred = autoencoder.predict(X_val)
            test_pred = autoencoder.predict(X_test)
            
            # Calculate reconstruction error
            train_error = np.mean(np.square(X_train - train_pred), axis=1)
            val_error = np.mean(np.square(X_val - val_pred), axis=1)
            test_error = np.mean(np.square(X_test - test_pred), axis=1)
            
            # Calculate threshold (95th percentile of training error)
            threshold = np.percentile(train_error, 95)
            
            # Log metrics
            mlflow.log_metrics({
                "train_reconstruction_error": np.mean(train_error),
                "val_reconstruction_error": np.mean(val_error),
                "test_reconstruction_error": np.mean(test_error),
                "anomaly_threshold": threshold
            })
            
            # Save model
            model_path = f"models/{model_name}"
            autoencoder.save(model_path)
            mlflow.tensorflow.log_model(autoencoder, "model")
            
            # Store results
            self.models[model_name] = autoencoder
            self.history[model_name] = history.history
            self.metrics[model_name] = {
                "threshold": threshold,
                "train_error": np.mean(train_error),
                "val_error": np.mean(val_error),
                "test_error": np.mean(test_error)
            }
            
            logger.info(f"Anomaly detection model {model_name} trained successfully")
            logger.info(f"Anomaly threshold: {threshold:.4f}")
            
            return {
                "model": autoencoder,
                "encoder": encoder,
                "decoder": decoder,
                "threshold": threshold,
                "history": history.history
            }
    
    def hyperparameter_optimization(self, X: np.ndarray, y: np.ndarray, 
                                  n_trials: int = 50) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        logger.info("Starting hyperparameter optimization")
        
        def objective(trial):
            # Suggest hyperparameters
            model_type = trial.suggest_categorical("model_type", ["lstm", "gru", "cnn_lstm"])
            sequence_length = trial.suggest_int("sequence_length", 12, 48)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            lstm_units = trial.suggest_int("lstm_units", 32, 128)
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
            
            # Update config
            config = TrainingConfig(
                model_type=model_type,
                sequence_length=sequence_length,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            # Create temporary training pipeline
            temp_pipeline = ModelTrainingPipeline(config)
            
            # Prepare data
            X_train, X_val, X_test, y_train, y_val, y_test = temp_pipeline.prepare_data(X, y)
            
            # Create model
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            if model_type == "lstm":
                model = temp_pipeline.create_lstm_model(input_shape)
            elif model_type == "gru":
                model = temp_pipeline.create_gru_model(input_shape)
            elif model_type == "cnn_lstm":
                model = temp_pipeline.create_cnn_lstm_model(input_shape)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20,  # Reduced for optimization
                batch_size=batch_size,
                verbose=0
            )
            
            # Evaluate
            val_pred = model.predict(X_val, verbose=0)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            
            return val_rmse
        
        # Run optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best validation RMSE: {study.best_value:.4f}")
        
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "study": study
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
    
    def save_models(self, base_path: str = "models"):
        """Save all trained models"""
        os.makedirs(base_path, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(base_path, model_name)
            model.save(model_path)
            logger.info(f"Model {model_name} saved to {model_path}")
    
    def load_models(self, base_path: str = "models"):
        """Load all trained models"""
        for model_name in os.listdir(base_path):
            model_path = os.path.join(base_path, model_name)
            if os.path.isdir(model_path):
                model = keras.models.load_model(model_path)
                self.models[model_name] = model
                logger.info(f"Model {model_name} loaded from {model_path}")
