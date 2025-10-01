"""
Equipment Failure Predictor
Predicts equipment failures and maintenance needs for smart meter infrastructure
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EquipmentFailurePredictor:
    """
    Predicts equipment failures and maintenance needs
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            },
            'logistic_regression': {
                'random_state': 42,
                'max_iter': 1000
            }
        }
        
        logger.info("EquipmentFailurePredictor initialized")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for failure prediction"""
        df = data.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df['timestamp'])
        
        # Equipment age features
        if 'installation_date' in df.columns:
            df['equipment_age_days'] = (df.index - pd.to_datetime(df['installation_date'])).dt.days
            df['equipment_age_years'] = df['equipment_age_days'] / 365.25
        
        # Usage intensity features
        if 'consumption' in df.columns:
            df['daily_consumption'] = df['consumption'].rolling(24).sum()
            df['consumption_intensity'] = df['consumption'] / df['consumption'].rolling(24).mean()
            df['peak_consumption'] = df['consumption'].rolling(24).max()
            df['consumption_variance'] = df['consumption'].rolling(24).var()
        
        # Temperature stress features
        if 'temperature' in df.columns:
            df['temperature_stress'] = np.abs(df['temperature'] - 20)  # Optimal around 20°C
            df['extreme_temperature'] = ((df['temperature'] < -10) | (df['temperature'] > 40)).astype(int)
            df['temperature_cycles'] = (df['temperature'].diff().abs() > 5).astype(int)
        
        # Voltage stress features
        if 'voltage' in df.columns:
            df['voltage_deviation'] = np.abs(df['voltage'] - 230)  # Optimal around 230V
            df['voltage_stress'] = (df['voltage'] < 200) | (df['voltage'] > 250)
            df['voltage_instability'] = df['voltage'].rolling(24).std()
        
        # Maintenance history features
        if 'last_maintenance' in df.columns:
            df['days_since_maintenance'] = (df.index - pd.to_datetime(df['last_maintenance'])).dt.days
            df['maintenance_overdue'] = (df['days_since_maintenance'] > 365).astype(int)
        
        # Failure history features
        if 'failure_count' in df.columns:
            df['failure_rate'] = df['failure_count'] / df['equipment_age_days']
            df['recent_failures'] = df['failure_count'].rolling(30).sum()
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_holiday'] = self._is_holiday(df.index)
        
        # Lag features for time series
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday']:
                for lag in [1, 7, 30]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
                # Rolling statistics
                for window in [7, 30, 90]:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
        
        return df
    
    def _is_holiday(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Check if dates are holidays (simplified)"""
        holidays = pd.Series(False, index=dates)
        
        # New Year, Christmas, etc. (simplified)
        holidays |= (dates.month == 1) & (dates.day == 1)
        holidays |= (dates.month == 12) & (dates.day == 25)
        
        return holidays.astype(int)
    
    def create_failure_labels(self, data: pd.DataFrame, failure_window_days: int = 30) -> pd.Series:
        """Create failure labels for training"""
        # This is a simplified approach - in practice, you'd have actual failure records
        labels = pd.Series(0, index=data.index)
        
        # Simulate failure events based on equipment age and stress
        if 'equipment_age_days' in data.columns:
            # Higher failure probability for older equipment
            age_factor = data['equipment_age_days'] / 365.25  # Age in years
            
            # Higher failure probability under stress
            stress_factor = 0
            if 'temperature_stress' in data.columns:
                stress_factor += data['temperature_stress'] / 20
            if 'voltage_deviation' in data.columns:
                stress_factor += data['voltage_deviation'] / 50
            
            # Combined failure probability
            failure_prob = 0.01 * age_factor + 0.005 * stress_factor
            
            # Generate random failures based on probability
            np.random.seed(42)  # For reproducibility
            random_values = np.random.random(len(data))
            failure_indices = random_values < failure_prob
            
            labels[failure_indices] = 1
        
        return labels
    
    def train_random_forest(self, data: pd.DataFrame, target_col: str = 'failure') -> Dict[str, Any]:
        """Train Random Forest model"""
        try:
            # Prepare features
            df_features = self.prepare_features(data)
            df_features = df_features.dropna()
            
            # Create target if not exists
            if target_col not in df_features.columns:
                df_features[target_col] = self.create_failure_labels(df_features)
            
            # Select features and target
            feature_cols = [col for col in df_features.columns if col != target_col]
            X = df_features[feature_cols].select_dtypes(include=[np.number])
            y = df_features[target_col]
            
            if X.empty:
                return {"error": "No numeric features available"}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(**self.model_configs['random_forest'])
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            
            # Feature importance
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            self.feature_importance['random_forest'] = feature_importance
            
            # Store model and scaler
            self.models['random_forest'] = model
            self.scalers['random_forest'] = scaler
            
            # Store metrics
            self.performance_metrics['random_forest'] = {
                'auc_score': auc_score,
                'classification_report': classification_rep,
                'feature_importance': feature_importance
            }
            
            logger.info(f"Random Forest trained - AUC: {auc_score:.4f}")
            
            return {
                'model_type': 'random_forest',
                'auc_score': auc_score,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': X.shape[1]
            }
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {str(e)}")
            raise
    
    def train_gradient_boosting(self, data: pd.DataFrame, target_col: str = 'failure') -> Dict[str, Any]:
        """Train Gradient Boosting model"""
        try:
            # Prepare features
            df_features = self.prepare_features(data)
            df_features = df_features.dropna()
            
            # Create target if not exists
            if target_col not in df_features.columns:
                df_features[target_col] = self.create_failure_labels(df_features)
            
            # Select features and target
            feature_cols = [col for col in df_features.columns if col != target_col]
            X = df_features[feature_cols].select_dtypes(include=[np.number])
            y = df_features[target_col]
            
            if X.empty:
                return {"error": "No numeric features available"}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = GradientBoostingClassifier(**self.model_configs['gradient_boosting'])
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            
            # Feature importance
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            self.feature_importance['gradient_boosting'] = feature_importance
            
            # Store model and scaler
            self.models['gradient_boosting'] = model
            self.scalers['gradient_boosting'] = scaler
            
            # Store metrics
            self.performance_metrics['gradient_boosting'] = {
                'auc_score': auc_score,
                'classification_report': classification_rep,
                'feature_importance': feature_importance
            }
            
            logger.info(f"Gradient Boosting trained - AUC: {auc_score:.4f}")
            
            return {
                'model_type': 'gradient_boosting',
                'auc_score': auc_score,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': X.shape[1]
            }
            
        except Exception as e:
            logger.error(f"Gradient Boosting training failed: {str(e)}")
            raise
    
    def train_logistic_regression(self, data: pd.DataFrame, target_col: str = 'failure') -> Dict[str, Any]:
        """Train Logistic Regression model"""
        try:
            # Prepare features
            df_features = self.prepare_features(data)
            df_features = df_features.dropna()
            
            # Create target if not exists
            if target_col not in df_features.columns:
                df_features[target_col] = self.create_failure_labels(df_features)
            
            # Select features and target
            feature_cols = [col for col in df_features.columns if col != target_col]
            X = df_features[feature_cols].select_dtypes(include=[np.number])
            y = df_features[target_col]
            
            if X.empty:
                return {"error": "No numeric features available"}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = LogisticRegression(**self.model_configs['logistic_regression'])
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            
            # Feature importance (coefficients)
            feature_importance = dict(zip(X.columns, np.abs(model.coef_[0])))
            self.feature_importance['logistic_regression'] = feature_importance
            
            # Store model and scaler
            self.models['logistic_regression'] = model
            self.scalers['logistic_regression'] = scaler
            
            # Store metrics
            self.performance_metrics['logistic_regression'] = {
                'auc_score': auc_score,
                'classification_report': classification_rep,
                'feature_importance': feature_importance
            }
            
            logger.info(f"Logistic Regression trained - AUC: {auc_score:.4f}")
            
            return {
                'model_type': 'logistic_regression',
                'auc_score': auc_score,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': X.shape[1]
            }
            
        except Exception as e:
            logger.error(f"Logistic Regression training failed: {str(e)}")
            raise
    
    def train_all_models(self, data: pd.DataFrame, target_col: str = 'failure') -> Dict[str, Any]:
        """Train all available models"""
        results = {}
        
        # Train each model
        for model_name in ['random_forest', 'gradient_boosting', 'logistic_regression']:
            try:
                if model_name == 'random_forest':
                    results[model_name] = self.train_random_forest(data, target_col)
                elif model_name == 'gradient_boosting':
                    results[model_name] = self.train_gradient_boosting(data, target_col)
                elif model_name == 'logistic_regression':
                    results[model_name] = self.train_logistic_regression(data, target_col)
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def predict_failure_risk(self, data: pd.DataFrame, model_name: str = 'random_forest') -> Dict[str, Any]:
        """Predict failure risk for equipment"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        try:
            # Prepare features
            df_features = self.prepare_features(data)
            df_features = df_features.dropna()
            
            # Select numeric features
            X = df_features.select_dtypes(include=[np.number])
            
            if X.empty:
                return {"error": "No numeric features available"}
            
            # Scale features
            scaler = self.scalers[model_name]
            X_scaled = scaler.transform(X)
            
            # Make predictions
            model = self.models[model_name]
            failure_probabilities = model.predict_proba(X_scaled)[:, 1]
            failure_predictions = model.predict(X_scaled)
            
            # Calculate risk levels
            risk_levels = []
            for prob in failure_probabilities:
                if prob < 0.1:
                    risk_levels.append('low')
                elif prob < 0.3:
                    risk_levels.append('medium')
                elif prob < 0.7:
                    risk_levels.append('high')
                else:
                    risk_levels.append('critical')
            
            return {
                'model_name': model_name,
                'failure_probabilities': failure_probabilities.tolist(),
                'failure_predictions': failure_predictions.tolist(),
                'risk_levels': risk_levels,
                'high_risk_count': sum(1 for level in risk_levels if level in ['high', 'critical']),
                'critical_risk_count': sum(1 for level in risk_levels if level == 'critical'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failure risk prediction failed: {str(e)}")
            raise
    
    def get_model_comparison(self) -> Dict[str, Any]:
        """Compare performance of all trained models"""
        if not self.performance_metrics:
            return {"error": "No models trained yet"}
        
        comparison = {}
        for model_name, metrics in self.performance_metrics.items():
            comparison[model_name] = {
                'auc_score': metrics['auc_score'],
                'precision': metrics['classification_report'].get('1', {}).get('precision', 0),
                'recall': metrics['classification_report'].get('1', {}).get('recall', 0),
                'f1_score': metrics['classification_report'].get('1', {}).get('f1-score', 0)
            }
        
        # Find best model
        best_model = max(comparison.keys(), key=lambda x: comparison[x]['auc_score'])
        comparison['best_model'] = best_model
        
        return comparison
    
    def save_models(self, filepath: str):
        """Save trained models"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'performance_metrics': self.performance_metrics,
            'feature_importance': self.feature_importance,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Equipment failure prediction models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.performance_metrics = model_data['performance_metrics']
        self.feature_importance = model_data['feature_importance']
        self.config = model_data['config']
        
        logger.info(f"Equipment failure prediction models loaded from {filepath}")
