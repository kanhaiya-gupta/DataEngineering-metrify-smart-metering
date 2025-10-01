"""
Feature Engineering Pipeline

This module implements automated feature engineering for smart meter data,
including time series features, statistical features, and domain-specific features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    lookback_window: int = 24  # hours
    forecast_horizon: int = 1   # hours
    include_weather: bool = True
    include_grid_status: bool = True
    include_time_features: bool = True
    include_lag_features: bool = True
    include_rolling_features: bool = True


class FeatureEngineeringPipeline:
    """
    Automated feature engineering pipeline for smart meter data
    
    This pipeline creates features for:
    - Time series forecasting
    - Anomaly detection
    - Grid optimization
    - Weather impact analysis
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.feature_columns = []
        self.scalers = {}
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()
        
        # Basic time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['year'] = df['timestamp'].dt.year
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Business time features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_holiday'] = self._is_holiday(df['timestamp'])
        df['is_peak_hours'] = df['hour'].between(17, 21).astype(int)
        df['is_night_hours'] = df['hour'].between(22, 6).astype(int)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create lagged features for time series"""
        df = df.copy()
        
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Difference features
        df[f'{target_col}_diff_1'] = df[target_col].diff(1)
        df[f'{target_col}_diff_24'] = df[target_col].diff(24)
        
        # Percentage change
        df[f'{target_col}_pct_change_1'] = df[target_col].pct_change(1)
        df[f'{target_col}_pct_change_24'] = df[target_col].pct_change(24)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create rolling window features"""
        df = df.copy()
        
        # Rolling statistics
        for window in [3, 6, 12, 24]:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
            df[f'{target_col}_rolling_median_{window}'] = df[target_col].rolling(window=window).median()
        
        # Exponential moving averages
        for alpha in [0.1, 0.3, 0.5]:
            df[f'{target_col}_ema_{alpha}'] = df[target_col].ewm(alpha=alpha).mean()
        
        return df
    
    def create_weather_features(self, df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Create weather-related features"""
        df = df.copy()
        
        # Merge weather data
        df = df.merge(weather_df, on='timestamp', how='left')
        
        # Weather features
        if 'temperature_c' in df.columns:
            df['temp_squared'] = df['temperature_c'] ** 2
            df['temp_cubed'] = df['temperature_c'] ** 3
            df['temp_rolling_mean_24'] = df['temperature_c'].rolling(24).mean()
        
        if 'humidity_percent' in df.columns:
            df['humidity_squared'] = df['humidity_percent'] ** 2
            df['comfort_index'] = self._calculate_comfort_index(
                df['temperature_c'], df['humidity_percent']
            )
        
        if 'wind_speed_mps' in df.columns:
            df['wind_speed_squared'] = df['wind_speed_mps'] ** 2
            df['wind_chill'] = self._calculate_wind_chill(
                df['temperature_c'], df['wind_speed_mps']
            )
        
        return df
    
    def create_grid_features(self, df: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
        """Create grid status related features"""
        df = df.copy()
        
        # Merge grid data
        df = df.merge(grid_df, on='timestamp', how='left')
        
        # Grid features
        if 'grid_load_mw' in df.columns:
            df['grid_load_normalized'] = df['grid_load_mw'] / df['grid_load_mw'].max()
            df['grid_load_rolling_mean_24'] = df['grid_load_mw'].rolling(24).mean()
        
        if 'frequency_hz' in df.columns:
            df['frequency_deviation'] = abs(df['frequency_hz'] - 50.0)
            df['frequency_stability'] = df['frequency_hz'].rolling(12).std()
        
        if 'voltage_kv' in df.columns:
            df['voltage_deviation'] = abs(df['voltage_kv'] - 230.0)
            df['voltage_stability'] = df['voltage_kv'].rolling(12).std()
        
        return df
    
    def create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features for energy data"""
        df = df.copy()
        
        # Energy consumption features
        if 'current_reading_kwh' in df.columns:
            # Consumption rate
            df['consumption_rate'] = df['current_reading_kwh'].diff()
            
            # Power factor features
            if 'power_factor' in df.columns:
                df['power_factor_squared'] = df['power_factor'] ** 2
                df['reactive_power'] = df['current_reading_kwh'] * np.sqrt(1 - df['power_factor'] ** 2)
            
            # Voltage and current features
            if 'voltage_v' in df.columns and 'amperage_a' in df.columns:
                df['apparent_power'] = df['voltage_v'] * df['amperage_a']
                df['power_efficiency'] = df['current_reading_kwh'] / df['apparent_power']
            
            # Temperature impact
            if 'temperature_c' in df.columns:
                df['temp_consumption_correlation'] = df['temperature_c'].rolling(24).corr(df['current_reading_kwh'])
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables"""
        df = df.copy()
        
        # Temperature and time interactions
        if 'temperature_c' in df.columns:
            df['temp_hour_interaction'] = df['temperature_c'] * df['hour']
            df['temp_weekend_interaction'] = df['temperature_c'] * df['is_weekend']
        
        # Weather and consumption interactions
        if 'temperature_c' in df.columns and 'current_reading_kwh' in df.columns:
            df['temp_consumption_interaction'] = df['temperature_c'] * df['current_reading_kwh']
        
        # Grid and consumption interactions
        if 'grid_load_mw' in df.columns and 'current_reading_kwh' in df.columns:
            df['grid_consumption_interaction'] = df['grid_load_mw'] * df['current_reading_kwh']
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale features for ML models"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        df = df.copy()
        
        # Select numeric columns for scaling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['timestamp', 'meter_id']]
        
        for col in numeric_cols:
            if fit:
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[[col]]).flatten()
                self.scalers[col] = scaler
            else:
                if col in self.scalers:
                    df[col] = self.scalers[col].transform(df[[col]]).flatten()
        
        return df
    
    def create_sequence_features(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM/RNN models"""
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(self.config.lookback_window, len(df) - self.config.forecast_horizon + 1):
            # Input sequence
            seq = df.iloc[i-self.config.lookback_window:i][self.feature_columns].values
            sequences.append(seq)
            
            # Target
            target = df.iloc[i:i+self.config.forecast_horizon][target_col].values
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def fit_transform(self, df: pd.DataFrame, weather_df: Optional[pd.DataFrame] = None, 
                    grid_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Fit the feature engineering pipeline and transform data"""
        logger.info("Starting feature engineering pipeline")
        
        # Create time features
        if self.config.include_time_features:
            df = self.create_time_features(df)
        
        # Create lag features
        if self.config.include_lag_features and 'current_reading_kwh' in df.columns:
            df = self.create_lag_features(df, 'current_reading_kwh')
        
        # Create rolling features
        if self.config.include_rolling_features and 'current_reading_kwh' in df.columns:
            df = self.create_rolling_features(df, 'current_reading_kwh')
        
        # Create weather features
        if self.config.include_weather and weather_df is not None:
            df = self.create_weather_features(df, weather_df)
        
        # Create grid features
        if self.config.include_grid_status and grid_df is not None:
            df = self.create_grid_features(df, grid_df)
        
        # Create domain features
        df = self.create_domain_features(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Scale features
        df = self.scale_features(df, fit=True)
        
        # Store feature columns
        self.feature_columns = [col for col in df.columns if col not in ['timestamp', 'meter_id']]
        
        logger.info(f"Feature engineering completed. Created {len(self.feature_columns)} features")
        return df
    
    def transform(self, df: pd.DataFrame, weather_df: Optional[pd.DataFrame] = None,
                 grid_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Transform new data using fitted pipeline"""
        logger.info("Transforming data with fitted feature engineering pipeline")
        
        # Apply same transformations without fitting
        if self.config.include_time_features:
            df = self.create_time_features(df)
        
        if self.config.include_lag_features and 'current_reading_kwh' in df.columns:
            df = self.create_lag_features(df, 'current_reading_kwh')
        
        if self.config.include_rolling_features and 'current_reading_kwh' in df.columns:
            df = self.create_rolling_features(df, 'current_reading_kwh')
        
        if self.config.include_weather and weather_df is not None:
            df = self.create_weather_features(df, weather_df)
        
        if self.config.include_grid_status and grid_df is not None:
            df = self.create_grid_features(df, grid_df)
        
        df = self.create_domain_features(df)
        df = self.create_interaction_features(df)
        df = self.scale_features(df, fit=False)
        
        return df
    
    def _is_holiday(self, timestamps: pd.Series) -> pd.Series:
        """Check if timestamps are holidays (simplified implementation)"""
        # This is a simplified implementation
        # In production, you would use a proper holiday calendar
        holidays = [
            '2024-01-01', '2024-03-29', '2024-04-01', '2024-05-01',
            '2024-05-09', '2024-05-20', '2024-10-03', '2024-12-25', '2024-12-26'
        ]
        
        holiday_dates = pd.to_datetime(holidays)
        return timestamps.dt.date.isin(holiday_dates.dt.date).astype(int)
    
    def _calculate_comfort_index(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate comfort index based on temperature and humidity"""
        # Simplified comfort index calculation
        return 0.5 * temp + 0.3 * humidity + 0.2 * (temp * humidity / 100)
    
    def _calculate_wind_chill(self, temp: pd.Series, wind_speed: pd.Series) -> pd.Series:
        """Calculate wind chill factor"""
        # Simplified wind chill calculation
        return temp - (wind_speed * 0.5)
