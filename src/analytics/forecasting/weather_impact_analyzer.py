"""
Weather Impact Analyzer
Analyzes the impact of weather conditions on energy consumption and grid load
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class WeatherImpactAnalyzer:
    """
    Analyzes weather impact on energy consumption and grid load
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.models = {}
        self.weather_impact_factors = {}
        self.performance_metrics = {}
        
        # Model configurations
        self.model_configs = {
            'linear': {
                'model': LinearRegression(),
                'features': ['temperature', 'humidity', 'wind_speed', 'pressure']
            },
            'ridge': {
                'model': Ridge(alpha=1.0),
                'features': ['temperature', 'humidity', 'wind_speed', 'pressure', 'temperature_squared']
            },
            'lasso': {
                'model': Lasso(alpha=0.1),
                'features': ['temperature', 'humidity', 'wind_speed', 'pressure', 'temperature_squared']
            },
            'random_forest': {
                'model': RandomForestRegressor(n_estimators=100, random_state=42),
                'features': ['temperature', 'humidity', 'wind_speed', 'pressure', 'temperature_squared', 'weather_category']
            }
        }
        
        logger.info("WeatherImpactAnalyzer initialized")
    
    def prepare_weather_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare weather features for analysis"""
        df = data.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df['timestamp'])
        
        # Basic weather features
        if 'temperature' in df.columns:
            df['temperature_squared'] = df['temperature'] ** 2
            df['temperature_cubed'] = df['temperature'] ** 3
            
            # Temperature categories
            df['temp_category'] = pd.cut(
                df['temperature'], 
                bins=[-np.inf, 0, 10, 20, 30, np.inf],
                labels=['very_cold', 'cold', 'mild', 'warm', 'hot']
            )
            
            # Heating and cooling degree days
            df['heating_degree_days'] = np.maximum(0, 18 - df['temperature'])
            df['cooling_degree_days'] = np.maximum(0, df['temperature'] - 25)
            
            # Temperature extremes
            df['extreme_cold'] = (df['temperature'] < -5).astype(int)
            df['extreme_heat'] = (df['temperature'] > 35).astype(int)
        
        # Humidity features
        if 'humidity' in df.columns:
            df['humidity_squared'] = df['humidity'] ** 2
            df['high_humidity'] = (df['humidity'] > 80).astype(int)
            df['low_humidity'] = (df['humidity'] < 30).astype(int)
        
        # Wind features
        if 'wind_speed' in df.columns:
            df['wind_speed_squared'] = df['wind_speed'] ** 2
            df['high_wind'] = (df['wind_speed'] > 15).astype(int)
            df['wind_chill'] = self._calculate_wind_chill(df.get('temperature', 20), df['wind_speed'])
        
        # Pressure features
        if 'pressure' in df.columns:
            df['pressure_change'] = df['pressure'].diff()
            df['low_pressure'] = (df['pressure'] < 1000).astype(int)
            df['high_pressure'] = (df['pressure'] > 1020).astype(int)
        
        # Weather interaction features
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['heat_index'] = self._calculate_heat_index(df['temperature'], df['humidity'])
            df['comfort_index'] = self._calculate_comfort_index(df['temperature'], df['humidity'], df.get('wind_speed', 0))
        
        # Seasonal features
        df['season'] = self._get_season(df.index)
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        df['is_holiday'] = self._is_holiday(df.index)
        
        # Weather category (simplified)
        df['weather_category'] = self._categorize_weather(df)
        
        # Lag features for weather
        weather_cols = ['temperature', 'humidity', 'wind_speed', 'pressure']
        for col in weather_cols:
            if col in df.columns:
                for lag in [1, 6, 12, 24]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Rolling weather statistics
        for col in weather_cols:
            if col in df.columns:
                for window in [6, 12, 24]:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
        
        return df
    
    def _calculate_wind_chill(self, temperature: pd.Series, wind_speed: pd.Series) -> pd.Series:
        """Calculate wind chill factor"""
        # Wind chill formula (simplified)
        wind_chill = 13.12 + 0.6215 * temperature - 11.37 * (wind_speed ** 0.16) + 0.3965 * temperature * (wind_speed ** 0.16)
        return wind_chill
    
    def _calculate_heat_index(self, temperature: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate heat index"""
        # Heat index formula (simplified)
        hi = -42.379 + 2.04901523 * temperature + 10.14333127 * humidity - 0.22475541 * temperature * humidity
        return hi
    
    def _calculate_comfort_index(self, temperature: pd.Series, humidity: pd.Series, wind_speed: pd.Series) -> pd.Series:
        """Calculate comfort index (0-100)"""
        # Simplified comfort index
        temp_score = 100 - abs(temperature - 22) * 2  # Optimal around 22°C
        humidity_score = 100 - abs(humidity - 50) * 0.5  # Optimal around 50%
        wind_score = 100 - wind_speed * 2  # Lower wind is better
        
        comfort = (temp_score + humidity_score + wind_score) / 3
        return np.clip(comfort, 0, 100)
    
    def _get_season(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Get season for dates"""
        seasons = pd.Series('unknown', index=dates)
        
        # Northern hemisphere seasons
        seasons[(dates.month >= 3) & (dates.month <= 5)] = 'spring'
        seasons[(dates.month >= 6) & (dates.month <= 8)] = 'summer'
        seasons[(dates.month >= 9) & (dates.month <= 11)] = 'autumn'
        seasons[(dates.month == 12) | (dates.month <= 2)] = 'winter'
        
        return seasons
    
    def _is_holiday(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Check if dates are holidays (simplified)"""
        holidays = pd.Series(False, index=dates)
        
        # New Year, Christmas, etc. (simplified)
        holidays |= (dates.month == 1) & (dates.day == 1)
        holidays |= (dates.month == 12) & (dates.day == 25)
        
        return holidays.astype(int)
    
    def _categorize_weather(self, df: pd.DataFrame) -> pd.Series:
        """Categorize weather conditions"""
        categories = pd.Series('normal', index=df.index)
        
        if 'temperature' in df.columns and 'humidity' in df.columns:
            # Hot and humid
            hot_humid = (df['temperature'] > 30) & (df['humidity'] > 70)
            categories[hot_humid] = 'hot_humid'
            
            # Cold and dry
            cold_dry = (df['temperature'] < 5) & (df['humidity'] < 40)
            categories[cold_dry] = 'cold_dry'
            
            # Mild conditions
            mild = (df['temperature'] >= 15) & (df['temperature'] <= 25) & (df['humidity'] >= 40) & (df['humidity'] <= 70)
            categories[mild] = 'mild'
        
        if 'wind_speed' in df.columns:
            # Windy conditions
            windy = df['wind_speed'] > 15
            categories[windy] = 'windy'
        
        return categories
    
    def analyze_weather_impact(self, data: pd.DataFrame, target_col: str = 'consumption') -> Dict[str, Any]:
        """Analyze weather impact on target variable"""
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Prepare features
        df_features = self.prepare_weather_features(data)
        df_features = df_features.dropna()
        
        if len(df_features) < 10:
            return {"error": "Insufficient data for weather impact analysis"}
        
        # Get weather features
        weather_features = [col for col in df_features.columns if col not in [target_col, 'timestamp']]
        X = df_features[weather_features].select_dtypes(include=[np.number])
        y = df_features[target_col]
        
        if X.empty:
            return {"error": "No numeric weather features found"}
        
        # Analyze with different models
        results = {}
        
        for model_name, model_config in self.model_configs.items():
            try:
                model = model_config['model']
                features = model_config['features']
                
                # Select available features
                available_features = [f for f in features if f in X.columns]
                if not available_features:
                    continue
                
                X_model = X[available_features]
                
                # Train model
                model.fit(X_model, y)
                
                # Evaluate model
                y_pred = model.predict(X_model)
                mae = mean_absolute_error(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(available_features, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    feature_importance = dict(zip(available_features, np.abs(model.coef_)))
                else:
                    feature_importance = {}
                
                results[model_name] = {
                    'mae': mae,
                    'mse': mse,
                    'r2': r2,
                    'rmse': np.sqrt(mse),
                    'feature_importance': feature_importance,
                    'features_used': available_features
                }
                
                # Store model
                self.models[model_name] = {
                    'model': model,
                    'features': available_features,
                    'scaler': StandardScaler().fit(X_model)
                }
                
            except Exception as e:
                logger.error(f"Weather impact analysis failed for {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        # Calculate weather impact factors
        self.weather_impact_factors = self._calculate_impact_factors(df_features, target_col)
        
        # Store performance metrics
        self.performance_metrics = results
        
        return {
            'model_results': results,
            'weather_impact_factors': self.weather_impact_factors,
            'best_model': max(results.keys(), key=lambda x: results[x].get('r2', 0)) if results else None
        }
    
    def _calculate_impact_factors(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Calculate weather impact factors"""
        factors = {}
        
        # Temperature impact
        if 'temperature' in df.columns:
            # Correlation with temperature
            temp_corr = df['temperature'].corr(df[target_col])
            factors['temperature_correlation'] = temp_corr
            
            # Temperature sensitivity (change in target per degree)
            temp_sensitivity = df[target_col].diff().corr(df['temperature'].diff())
            factors['temperature_sensitivity'] = temp_sensitivity
            
            # Heating/cooling impact
            if 'heating_degree_days' in df.columns:
                heating_corr = df['heating_degree_days'].corr(df[target_col])
                factors['heating_impact'] = heating_corr
            
            if 'cooling_degree_days' in df.columns:
                cooling_corr = df['cooling_degree_days'].corr(df[target_col])
                factors['cooling_impact'] = cooling_corr
        
        # Humidity impact
        if 'humidity' in df.columns:
            humidity_corr = df['humidity'].corr(df[target_col])
            factors['humidity_correlation'] = humidity_corr
        
        # Wind impact
        if 'wind_speed' in df.columns:
            wind_corr = df['wind_speed'].corr(df[target_col])
            factors['wind_correlation'] = wind_corr
        
        # Pressure impact
        if 'pressure' in df.columns:
            pressure_corr = df['pressure'].corr(df[target_col])
            factors['pressure_correlation'] = pressure_corr
        
        # Seasonal impact
        if 'season' in df.columns:
            seasonal_impact = {}
            for season in df['season'].unique():
                if pd.notna(season):
                    season_data = df[df['season'] == season]
                    seasonal_impact[season] = {
                        'mean_consumption': season_data[target_col].mean(),
                        'std_consumption': season_data[target_col].std()
                    }
            factors['seasonal_impact'] = seasonal_impact
        
        # Weather category impact
        if 'weather_category' in df.columns:
            category_impact = {}
            for category in df['weather_category'].unique():
                if pd.notna(category):
                    category_data = df[df['weather_category'] == category]
                    category_impact[category] = {
                        'mean_consumption': category_data[target_col].mean(),
                        'std_consumption': category_data[target_col].std()
                    }
            factors['weather_category_impact'] = category_impact
        
        return factors
    
    def predict_weather_impact(self, weather_data: pd.DataFrame, model_name: str = 'random_forest') -> Dict[str, Any]:
        """Predict impact of weather on energy consumption"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model_info = self.models[model_name]
        model = model_info['model']
        features = model_info['features']
        scaler = model_info['scaler']
        
        # Prepare weather features
        df_features = self.prepare_weather_features(weather_data)
        
        # Select features
        available_features = [f for f in features if f in df_features.columns]
        X = df_features[available_features].select_dtypes(include=[np.number])
        
        if X.empty:
            raise ValueError("No numeric features available for prediction")
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        return {
            'predictions': predictions.tolist(),
            'model_name': model_name,
            'features_used': available_features,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_weather_sensitivity_analysis(self) -> Dict[str, Any]:
        """Get weather sensitivity analysis"""
        if not self.weather_impact_factors:
            return {"error": "No weather impact analysis available"}
        
        sensitivity = {}
        
        # Temperature sensitivity
        if 'temperature_correlation' in self.weather_impact_factors:
            temp_corr = self.weather_impact_factors['temperature_correlation']
            sensitivity['temperature'] = {
                'correlation': temp_corr,
                'impact_level': 'high' if abs(temp_corr) > 0.5 else 'medium' if abs(temp_corr) > 0.3 else 'low',
                'direction': 'positive' if temp_corr > 0 else 'negative'
            }
        
        # Other weather factors
        for factor in ['humidity', 'wind_speed', 'pressure']:
            corr_key = f'{factor}_correlation'
            if corr_key in self.weather_impact_factors:
                corr = self.weather_impact_factors[corr_key]
                sensitivity[factor] = {
                    'correlation': corr,
                    'impact_level': 'high' if abs(corr) > 0.5 else 'medium' if abs(corr) > 0.3 else 'low',
                    'direction': 'positive' if corr > 0 else 'negative'
                }
        
        return sensitivity
    
    def get_seasonal_patterns(self) -> Dict[str, Any]:
        """Get seasonal consumption patterns"""
        if 'seasonal_impact' not in self.weather_impact_factors:
            return {"error": "No seasonal analysis available"}
        
        seasonal_data = self.weather_impact_factors['seasonal_impact']
        
        # Find peak and low seasons
        season_means = {season: data['mean_consumption'] for season, data in seasonal_data.items()}
        peak_season = max(season_means.keys(), key=lambda x: season_means[x])
        low_season = min(season_means.keys(), key=lambda x: season_means[x])
        
        return {
            'seasonal_data': seasonal_data,
            'peak_season': peak_season,
            'low_season': low_season,
            'seasonal_variation': (season_means[peak_season] - season_means[low_season]) / season_means[low_season]
        }
    
    def get_weather_category_impact(self) -> Dict[str, Any]:
        """Get weather category impact analysis"""
        if 'weather_category_impact' not in self.weather_impact_factors:
            return {"error": "No weather category analysis available"}
        
        category_data = self.weather_impact_factors['weather_category_impact']
        
        # Find most and least impactful weather categories
        category_means = {category: data['mean_consumption'] for category, data in category_data.items()}
        highest_impact = max(category_means.keys(), key=lambda x: category_means[x])
        lowest_impact = min(category_means.keys(), key=lambda x: category_means[x])
        
        return {
            'category_data': category_data,
            'highest_impact_category': highest_impact,
            'lowest_impact_category': lowest_impact,
            'impact_variation': (category_means[highest_impact] - category_means[lowest_impact]) / category_means[lowest_impact]
        }
    
    def export_analysis_report(self, filepath: str):
        """Export weather impact analysis report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'weather_impact_factors': self.weather_impact_factors,
            'model_performance': self.performance_metrics,
            'sensitivity_analysis': self.get_weather_sensitivity_analysis(),
            'seasonal_patterns': self.get_seasonal_patterns(),
            'weather_category_impact': self.get_weather_category_impact()
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Weather impact analysis report exported to {filepath}")
    
    def save_models(self, filepath: str):
        """Save trained models"""
        model_data = {
            'models': {name: info for name, info in self.models.items() if name != 'model'},
            'weather_impact_factors': self.weather_impact_factors,
            'performance_metrics': self.performance_metrics,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Weather impact models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.weather_impact_factors = model_data['weather_impact_factors']
        self.performance_metrics = model_data['performance_metrics']
        self.config = model_data['config']
        
        logger.info(f"Weather impact models loaded from {filepath}")
