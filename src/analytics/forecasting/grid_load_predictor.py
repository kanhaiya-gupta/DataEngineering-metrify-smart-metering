"""
Grid Load Predictor
Advanced grid load forecasting for energy grid optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class GridLoadPredictor:
    """
    Advanced grid load forecasting with multiple model support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
        # Model configurations
        self.model_configs = {
            'lstm': {
                'sequence_length': 24,
                'forecast_horizon': 24,
                'features': ['grid_load', 'consumption', 'temperature', 'hour', 'day_of_week']
            },
            'prophet': {
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': True
            },
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.1
            },
            'ensemble': {
                'models': ['lstm', 'prophet', 'xgboost'],
                'weights': [0.4, 0.3, 0.3]
            }
        }
        
        logger.info("GridLoadPredictor initialized")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for grid load forecasting"""
        df = data.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df['timestamp'])
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_holiday'] = self._is_holiday(df.index)
        
        # Grid load features
        if 'grid_load' in df.columns:
            # Lag features
            for lag in [1, 2, 3, 6, 12, 24, 48]:
                df[f'grid_load_lag_{lag}'] = df['grid_load'].shift(lag)
            
            # Rolling statistics
            for window in [3, 6, 12, 24, 48]:
                df[f'grid_load_rolling_mean_{window}'] = df['grid_load'].rolling(window).mean()
                df[f'grid_load_rolling_std_{window}'] = df['grid_load'].rolling(window).std()
                df[f'grid_load_rolling_max_{window}'] = df['grid_load'].rolling(window).max()
                df[f'grid_load_rolling_min_{window}'] = df['grid_load'].rolling(window).min()
            
            # Difference features
            df['grid_load_diff_1'] = df['grid_load'].diff(1)
            df['grid_load_diff_24'] = df['grid_load'].diff(24)
            df['grid_load_pct_change'] = df['grid_load'].pct_change()
        
        # Consumption features (if available)
        if 'consumption' in df.columns:
            df['consumption_lag_24'] = df['consumption'].shift(24)
            df['consumption_rolling_mean_24'] = df['consumption'].rolling(24).mean()
            df['total_consumption'] = df['consumption'].rolling(24).sum()
        
        # Weather features (if available)
        if 'temperature' in df.columns:
            df['temperature_lag_1'] = df['temperature'].shift(1)
            df['temperature_rolling_mean_24'] = df['temperature'].rolling(24).mean()
            df['temperature_diff'] = df['temperature'].diff(1)
            
            # Temperature impact features
            df['temp_impact'] = df['temperature'] * df.get('consumption', 0)
            df['heating_demand'] = np.maximum(0, 18 - df['temperature'])  # Heating below 18°C
            df['cooling_demand'] = np.maximum(0, df['temperature'] - 25)  # Cooling above 25°C
        
        # Grid stability features
        if 'grid_load' in df.columns:
            df['load_factor'] = df['grid_load'] / df['grid_load'].rolling(24).max()
            df['load_variance'] = df['grid_load'].rolling(24).var()
            df['peak_load'] = (df['grid_load'] > df['grid_load'].rolling(24).quantile(0.9)).astype(int)
        
        # Economic features
        df['hourly_price_impact'] = self._calculate_hourly_price_impact(df.index)
        df['demand_response_potential'] = self._calculate_demand_response_potential(df)
        
        return df
    
    def _is_holiday(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Check if dates are holidays (simplified)"""
        # This is a simplified holiday detection
        # In practice, you'd use a proper holiday calendar
        holidays = pd.Series(False, index=dates)
        
        # New Year, Christmas, etc. (simplified)
        holidays |= (dates.month == 1) & (dates.day == 1)
        holidays |= (dates.month == 12) & (dates.day == 25)
        
        return holidays.astype(int)
    
    def _calculate_hourly_price_impact(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Calculate hourly price impact on grid load"""
        # Simplified price impact calculation
        # In practice, you'd use actual electricity price data
        price_impact = pd.Series(1.0, index=dates)
        
        # Higher prices during peak hours (8-20)
        peak_hours = (dates.hour >= 8) & (dates.hour <= 20)
        price_impact[peak_hours] = 1.2
        
        # Lower prices during night hours (22-6)
        night_hours = (dates.hour >= 22) | (dates.hour <= 6)
        price_impact[night_hours] = 0.8
        
        return price_impact
    
    def _calculate_demand_response_potential(self, df: pd.DataFrame) -> pd.Series:
        """Calculate demand response potential"""
        if 'grid_load' not in df.columns:
            return pd.Series(0.0, index=df.index)
        
        # Calculate potential for demand response based on load variability
        load_std = df['grid_load'].rolling(24).std()
        load_mean = df['grid_load'].rolling(24).mean()
        
        # Higher variability = higher demand response potential
        demand_response = load_std / (load_mean + 1e-6)
        
        return demand_response.fillna(0)
    
    def create_sequences(self, data: pd.DataFrame, sequence_length: int, 
                        target_col: str = 'grid_load') -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM models"""
        features = data.drop(columns=[target_col]).select_dtypes(include=[np.number])
        target = data[target_col].values
        
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(features.iloc[i-sequence_length:i].values)
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def train_lstm_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train LSTM model for grid load forecasting"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            from sklearn.preprocessing import MinMaxScaler
            
            # Prepare data
            df_features = self.prepare_features(data)
            df_features = df_features.dropna()
            
            # Scale features
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df_features.select_dtypes(include=[np.number]))
            
            # Create sequences
            sequence_length = self.model_configs['lstm']['sequence_length']
            X, y = self.create_sequences(
                pd.DataFrame(scaled_data, index=df_features.index, 
                           columns=df_features.select_dtypes(include=[np.number]).columns),
                sequence_length
            )
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build model
            model = Sequential([
                LSTM(100, return_sequences=True, input_shape=(sequence_length, X.shape[2])),
                BatchNormalization(),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                BatchNormalization(),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=0,
                callbacks=[
                    # Early stopping would be added here
                ]
            )
            
            # Evaluate model
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store model and metrics
            self.models['lstm'] = {
                'model': model,
                'scaler': scaler,
                'sequence_length': sequence_length
            }
            
            self.performance_metrics['lstm'] = {
                'mae': mae,
                'mse': mse,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
            
            logger.info(f"LSTM model trained - MAE: {mae:.4f}, R2: {r2:.4f}")
            
            return {
                'model_type': 'lstm',
                'metrics': self.performance_metrics['lstm'],
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            logger.error(f"LSTM training failed: {str(e)}")
            raise
    
    def train_prophet_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train Prophet model for grid load forecasting"""
        try:
            from prophet import Prophet
            
            # Prepare data for Prophet
            df_features = self.prepare_features(data)
            prophet_data = df_features[['grid_load']].reset_index()
            prophet_data.columns = ['ds', 'y']
            prophet_data = prophet_data.dropna()
            
            # Add additional regressors if available
            model = Prophet(
                yearly_seasonality=self.model_configs['prophet']['yearly_seasonality'],
                weekly_seasonality=self.model_configs['prophet']['weekly_seasonality'],
                daily_seasonality=self.model_configs['prophet']['daily_seasonality']
            )
            
            # Add temperature as regressor if available
            if 'temperature' in df_features.columns:
                temp_data = df_features[['temperature']].reset_index()
                temp_data.columns = ['ds', 'temperature']
                prophet_data = prophet_data.merge(temp_data, on='ds', how='left')
                model.add_regressor('temperature')
            
            # Add consumption as regressor if available
            if 'consumption' in df_features.columns:
                cons_data = df_features[['consumption']].reset_index()
                cons_data.columns = ['ds', 'consumption']
                prophet_data = prophet_data.merge(cons_data, on='ds', how='left')
                model.add_regressor('consumption')
            
            model.fit(prophet_data)
            
            # Evaluate model
            future = model.make_future_dataframe(periods=24, freq='H')
            
            # Add regressors to future data
            if 'temperature' in prophet_data.columns:
                future['temperature'] = prophet_data['temperature'].iloc[-1]  # Use last value
            if 'consumption' in prophet_data.columns:
                future['consumption'] = prophet_data['consumption'].iloc[-1]  # Use last value
            
            forecast = model.predict(future)
            
            # Get predictions for evaluation
            y_pred = forecast['yhat'].tail(24).values
            y_actual = prophet_data['y'].tail(24).values
            
            mae = mean_absolute_error(y_actual, y_pred)
            mse = mean_squared_error(y_actual, y_pred)
            r2 = r2_score(y_actual, y_pred)
            
            # Store model and metrics
            self.models['prophet'] = {
                'model': model,
                'forecast': forecast
            }
            
            self.performance_metrics['prophet'] = {
                'mae': mae,
                'mse': mse,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
            
            logger.info(f"Prophet model trained - MAE: {mae:.4f}, R2: {r2:.4f}")
            
            return {
                'model_type': 'prophet',
                'metrics': self.performance_metrics['prophet'],
                'training_samples': len(prophet_data)
            }
            
        except Exception as e:
            logger.error(f"Prophet training failed: {str(e)}")
            raise
    
    def train_xgboost_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train XGBoost model for grid load forecasting"""
        try:
            import xgboost as xgb
            from sklearn.model_selection import train_test_split
            
            # Prepare features
            df_features = self.prepare_features(data)
            df_features = df_features.dropna()
            
            # Prepare features and target
            feature_cols = [col for col in df_features.columns if col != 'grid_load']
            X = df_features[feature_cols]
            y = df_features['grid_load']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train XGBoost model
            model = xgb.XGBRegressor(
                n_estimators=self.model_configs['xgboost']['n_estimators'],
                max_depth=self.model_configs['xgboost']['max_depth'],
                learning_rate=self.model_configs['xgboost']['learning_rate'],
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            self.feature_importance['xgboost'] = feature_importance
            
            # Store model and metrics
            self.models['xgboost'] = {
                'model': model,
                'feature_cols': feature_cols
            }
            
            self.performance_metrics['xgboost'] = {
                'mae': mae,
                'mse': mse,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
            
            logger.info(f"XGBoost model trained - MAE: {mae:.4f}, R2: {r2:.4f}")
            
            return {
                'model_type': 'xgboost',
                'metrics': self.performance_metrics['xgboost'],
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {str(e)}")
            raise
    
    def train_ensemble_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train ensemble model combining multiple models"""
        try:
            # Train individual models first
            lstm_result = self.train_lstm_model(data)
            prophet_result = self.train_prophet_model(data)
            xgboost_result = self.train_xgboost_model(data)
            
            # Create ensemble
            ensemble_weights = self.model_configs['ensemble']['weights']
            
            self.models['ensemble'] = {
                'models': ['lstm', 'prophet', 'xgboost'],
                'weights': ensemble_weights
            }
            
            # Calculate ensemble metrics (simplified)
            ensemble_mae = np.average([
                self.performance_metrics['lstm']['mae'],
                self.performance_metrics['prophet']['mae'],
                self.performance_metrics['xgboost']['mae']
            ], weights=ensemble_weights)
            
            ensemble_r2 = np.average([
                self.performance_metrics['lstm']['r2'],
                self.performance_metrics['prophet']['r2'],
                self.performance_metrics['xgboost']['r2']
            ], weights=ensemble_weights)
            
            self.performance_metrics['ensemble'] = {
                'mae': ensemble_mae,
                'r2': ensemble_r2,
                'rmse': ensemble_mae  # Simplified
            }
            
            logger.info(f"Ensemble model trained - MAE: {ensemble_mae:.4f}, R2: {ensemble_r2:.4f}")
            
            return {
                'model_type': 'ensemble',
                'metrics': self.performance_metrics['ensemble'],
                'component_models': ['lstm', 'prophet', 'xgboost']
            }
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {str(e)}")
            raise
    
    def train_all_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train all available models"""
        results = {}
        
        # Train each model
        for model_name in ['lstm', 'prophet', 'xgboost', 'ensemble']:
            try:
                if model_name == 'lstm':
                    results[model_name] = self.train_lstm_model(data)
                elif model_name == 'prophet':
                    results[model_name] = self.train_prophet_model(data)
                elif model_name == 'xgboost':
                    results[model_name] = self.train_xgboost_model(data)
                elif model_name == 'ensemble':
                    results[model_name] = self.train_ensemble_model(data)
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def forecast(self, model_name: str, data: pd.DataFrame, 
                horizon: int = 24) -> Dict[str, Any]:
        """Make forecast using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        try:
            if model_name == 'lstm':
                return self._forecast_lstm(horizon)
            elif model_name == 'prophet':
                return self._forecast_prophet(horizon)
            elif model_name == 'xgboost':
                return self._forecast_xgboost(data, horizon)
            elif model_name == 'ensemble':
                return self._forecast_ensemble(data, horizon)
        except Exception as e:
            logger.error(f"Forecast failed for {model_name}: {str(e)}")
            raise
    
    def _forecast_lstm(self, horizon: int) -> Dict[str, Any]:
        """LSTM forecast"""
        model_info = self.models['lstm']
        model = model_info['model']
        scaler = model_info['scaler']
        sequence_length = model_info['sequence_length']
        
        # This is a simplified forecast - in practice, you'd need recent data
        # to create the input sequence
        forecast = np.random.randn(horizon) * 0.1 + 1000  # Placeholder
        
        return {
            'forecast': forecast.tolist(),
            'model_type': 'lstm',
            'horizon': horizon,
            'confidence_interval': None
        }
    
    def _forecast_prophet(self, horizon: int) -> Dict[str, Any]:
        """Prophet forecast"""
        model = self.models['prophet']['model']
        future = model.make_future_dataframe(periods=horizon, freq='H')
        forecast = model.predict(future)
        
        return {
            'forecast': forecast['yhat'].tail(horizon).tolist(),
            'model_type': 'prophet',
            'horizon': horizon,
            'confidence_interval': {
                'lower': forecast['yhat_lower'].tail(horizon).tolist(),
                'upper': forecast['yhat_upper'].tail(horizon).tolist()
            }
        }
    
    def _forecast_xgboost(self, data: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """XGBoost forecast"""
        model_info = self.models['xgboost']
        model = model_info['model']
        feature_cols = model_info['feature_cols']
        
        # Prepare features for forecast
        df_features = self.prepare_features(data)
        df_features = df_features.dropna()
        
        # Use last available data for forecast
        last_features = df_features[feature_cols].iloc[-1:].values
        
        # Generate forecast (simplified)
        forecast = []
        for _ in range(horizon):
            pred = model.predict(last_features)[0]
            forecast.append(pred)
            # Update features for next prediction (simplified)
            last_features[0][0] = pred  # Update grid_load feature
        
        return {
            'forecast': forecast,
            'model_type': 'xgboost',
            'horizon': horizon,
            'confidence_interval': None
        }
    
    def _forecast_ensemble(self, data: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Ensemble forecast"""
        ensemble_info = self.models['ensemble']
        weights = ensemble_info['weights']
        
        # Get forecasts from individual models
        lstm_forecast = self._forecast_lstm(horizon)['forecast']
        prophet_forecast = self._forecast_prophet(horizon)['forecast']
        xgboost_forecast = self._forecast_xgboost(data, horizon)['forecast']
        
        # Combine forecasts
        ensemble_forecast = []
        for i in range(horizon):
            weighted_pred = (
                weights[0] * lstm_forecast[i] +
                weights[1] * prophet_forecast[i] +
                weights[2] * xgboost_forecast[i]
            )
            ensemble_forecast.append(weighted_pred)
        
        return {
            'forecast': ensemble_forecast,
            'model_type': 'ensemble',
            'horizon': horizon,
            'confidence_interval': None,
            'component_forecasts': {
                'lstm': lstm_forecast,
                'prophet': prophet_forecast,
                'xgboost': xgboost_forecast
            }
        }
    
    def get_model_comparison(self) -> Dict[str, Any]:
        """Compare performance of all trained models"""
        if not self.performance_metrics:
            return {"error": "No models trained yet"}
        
        comparison = {}
        for model_name, metrics in self.performance_metrics.items():
            comparison[model_name] = {
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2']
            }
        
        # Find best model
        best_model = min(comparison.keys(), key=lambda x: comparison[x]['mae'])
        comparison['best_model'] = best_model
        
        return comparison
    
    def save_models(self, filepath: str):
        """Save trained models"""
        model_data = {
            'models': {},
            'performance_metrics': self.performance_metrics,
            'feature_importance': self.feature_importance,
            'config': self.config
        }
        
        # Save non-TensorFlow models
        for name, model_info in self.models.items():
            if name != 'lstm':  # Skip TensorFlow models
                model_data['models'][name] = model_info
        
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.performance_metrics = model_data['performance_metrics']
        self.feature_importance = model_data['feature_importance']
        self.config = model_data['config']
        
        logger.info(f"Models loaded from {filepath}")
