"""
Data Preprocessing Pipeline

This module implements data preprocessing for smart meter data,
including cleaning, validation, and preparation for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing"""
    handle_missing: str = "interpolate"  # interpolate, drop, impute
    imputation_method: str = "knn"  # simple, knn
    scaling_method: str = "standard"  # standard, minmax, robust
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: float = 3.0
    feature_selection: bool = True
    n_features: int = 50
    pca_components: Optional[int] = None
    remove_duplicates: bool = True
    time_series_resample: bool = True
    resample_freq: str = "1H"


class DataPreprocessingPipeline:
    """
    Comprehensive data preprocessing pipeline for smart meter data
    
    Handles:
    - Missing data imputation
    - Outlier detection and treatment
    - Feature scaling and normalization
    - Feature selection and dimensionality reduction
    - Time series resampling and alignment
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.scalers = {}
        self.imputers = {}
        self.feature_selectors = {}
        self.pca = None
        self.is_fitted = False
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw data"""
        logger.info("Starting data cleaning")
        
        df = df.copy()
        original_shape = df.shape
        
        # Remove duplicates
        if self.config.remove_duplicates:
            df = df.drop_duplicates()
            logger.info(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        # Remove rows with invalid timestamps
        if 'timestamp' in df.columns:
            df = df.dropna(subset=['timestamp'])
            df = df[df['timestamp'] >= pd.Timestamp('2020-01-01')]  # Reasonable date range
        
        # Clean numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Replace infinite values with NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Remove rows where all numeric values are NaN
            if df[col].isna().all():
                logger.warning(f"Column {col} has all NaN values")
        
        logger.info(f"Data cleaning completed. Shape: {original_shape} -> {df.shape}")
        return df
    
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data"""
        logger.info("Handling missing data")
        
        df = df.copy()
        missing_before = df.isnull().sum().sum()
        
        if self.config.handle_missing == "drop":
            # Drop rows with any missing values
            df = df.dropna()
            logger.info(f"Dropped rows with missing values. Remaining: {df.shape[0]} rows")
            
        elif self.config.handle_missing == "interpolate":
            # Interpolate missing values for time series
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
                df = df.interpolate(method='time')
                df = df.reset_index()
            else:
                df = df.interpolate()
            logger.info("Interpolated missing values")
            
        elif self.config.handle_missing == "impute":
            # Use imputation methods
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if self.config.imputation_method == "simple":
                imputer = SimpleImputer(strategy='mean')
            elif self.config.imputation_method == "knn":
                imputer = KNNImputer(n_neighbors=5)
            else:
                raise ValueError(f"Unsupported imputation method: {self.config.imputation_method}")
            
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            self.imputers['numeric'] = imputer
            logger.info(f"Imputed missing values using {self.config.imputation_method}")
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing data handling completed. Missing values: {missing_before} -> {missing_after}")
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers"""
        logger.info("Detecting outliers")
        
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        
        for col in numeric_cols:
            if col in ['timestamp', 'meter_id']:
                continue
                
            if self.config.outlier_method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif self.config.outlier_method == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > self.config.outlier_threshold
                
            else:
                logger.warning(f"Unsupported outlier method: {self.config.outlier_method}")
                continue
            
            outlier_counts[col] = outliers.sum()
            
            # Cap outliers instead of removing them
            if outliers.any():
                df.loc[outliers, col] = df[col].clip(
                    lower=df[col].quantile(0.01),
                    upper=df[col].quantile(0.99)
                )
        
        total_outliers = sum(outlier_counts.values())
        logger.info(f"Outlier detection completed. Total outliers handled: {total_outliers}")
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale and normalize features"""
        logger.info("Scaling features")
        
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['timestamp', 'meter_id']]
        
        if self.config.scaling_method == "standard":
            scaler = StandardScaler()
        elif self.config.scaling_method == "minmax":
            scaler = MinMaxScaler()
        elif self.config.scaling_method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {self.config.scaling_method}")
        
        if fit:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.scalers['numeric'] = scaler
        else:
            if 'numeric' in self.scalers:
                df[numeric_cols] = self.scalers['numeric'].transform(df[numeric_cols])
        
        logger.info(f"Feature scaling completed using {self.config.scaling_method}")
        return df
    
    def select_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """Select most relevant features"""
        if not self.config.feature_selection or target_col is None:
            return df
        
        logger.info("Selecting features")
        
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in [target_col, 'timestamp', 'meter_id']]
        
        if len(feature_cols) <= self.config.n_features:
            logger.info(f"Number of features ({len(feature_cols)}) <= target ({self.config.n_features}). No selection needed.")
            return df
        
        # Select features using statistical tests
        selector = SelectKBest(score_func=f_regression, k=self.config.n_features)
        
        X = df[feature_cols].fillna(0)  # Fill any remaining NaN values
        y = df[target_col].fillna(0)
        
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        
        # Keep only selected features plus metadata columns
        keep_cols = selected_features + [target_col, 'timestamp', 'meter_id']
        keep_cols = [col for col in keep_cols if col in df.columns]
        
        df = df[keep_cols]
        self.feature_selectors['numeric'] = selector
        
        logger.info(f"Feature selection completed. Selected {len(selected_features)} features")
        return df
    
    def apply_pca(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Principal Component Analysis"""
        if self.config.pca_components is None:
            return df
        
        logger.info("Applying PCA")
        
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['timestamp', 'meter_id']]
        
        if len(feature_cols) <= self.config.pca_components:
            logger.info(f"Number of features ({len(feature_cols)}) <= PCA components ({self.config.pca_components}). Skipping PCA.")
            return df
        
        # Apply PCA
        pca = PCA(n_components=self.config.pca_components)
        X_pca = pca.fit_transform(df[feature_cols])
        
        # Create new DataFrame with PCA components
        pca_cols = [f'pca_{i}' for i in range(self.config.pca_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
        
        # Combine PCA components with metadata
        metadata_cols = ['timestamp', 'meter_id']
        metadata_cols = [col for col in metadata_cols if col in df.columns]
        
        df_pca = pd.concat([df[metadata_cols], pca_df], axis=1)
        
        self.pca = pca
        logger.info(f"PCA completed. Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
        
        return df_pca
    
    def resample_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample time series data"""
        if not self.config.time_series_resample or 'timestamp' not in df.columns:
            return df
        
        logger.info("Resampling time series data")
        
        df = df.copy()
        df = df.set_index('timestamp')
        
        # Resample to specified frequency
        df_resampled = df.resample(self.config.resample_freq).agg({
            col: 'mean' if df[col].dtype in ['float64', 'int64'] else 'first'
            for col in df.columns
        })
        
        df_resampled = df_resampled.reset_index()
        
        logger.info(f"Time series resampling completed. New frequency: {self.config.resample_freq}")
        return df_resampled
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate preprocessed data"""
        logger.info("Validating preprocessed data")
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "shape": df.shape,
            "missing_values": df.isnull().sum().sum(),
            "infinite_values": np.isinf(df.select_dtypes(include=[np.number])).sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
            "data_types": df.dtypes.to_dict(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "validation_passed": True
        }
        
        # Check for issues
        issues = []
        
        if validation_results["missing_values"] > 0:
            issues.append(f"Found {validation_results['missing_values']} missing values")
        
        if validation_results["infinite_values"] > 0:
            issues.append(f"Found {validation_results['infinite_values']} infinite values")
        
        if validation_results["duplicate_rows"] > 0:
            issues.append(f"Found {validation_results['duplicate_rows']} duplicate rows")
        
        if len(issues) > 0:
            validation_results["issues"] = issues
            validation_results["validation_passed"] = False
            logger.warning(f"Data validation issues: {issues}")
        else:
            logger.info("Data validation passed successfully")
        
        return validation_results
    
    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fit the preprocessing pipeline and transform data"""
        logger.info("Starting data preprocessing pipeline")
        
        # Clean data
        df = self.clean_data(df)
        
        # Handle missing data
        df = self.handle_missing_data(df)
        
        # Detect and handle outliers
        df = self.detect_outliers(df)
        
        # Scale features
        df = self.scale_features(df, fit=True)
        
        # Select features
        df = self.select_features(df, target_col)
        
        # Apply PCA if configured
        df = self.apply_pca(df)
        
        # Resample time series if configured
        df = self.resample_time_series(df)
        
        # Validate data
        validation_results = self.validate_data(df)
        
        self.is_fitted = True
        
        logger.info("Data preprocessing pipeline completed successfully")
        return df, validation_results
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit_transform() first.")
        
        logger.info("Transforming data with fitted preprocessing pipeline")
        
        # Clean data
        df = self.clean_data(df)
        
        # Handle missing data (using fitted imputers)
        df = self.handle_missing_data(df)
        
        # Detect and handle outliers
        df = self.detect_outliers(df)
        
        # Scale features (using fitted scalers)
        df = self.scale_features(df, fit=False)
        
        # Apply PCA if configured (using fitted PCA)
        if self.pca is not None:
            df = self.apply_pca(df)
        
        # Resample time series if configured
        df = self.resample_time_series(df)
        
        logger.info("Data transformation completed")
        return df
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about the preprocessing pipeline"""
        return {
            "is_fitted": self.is_fitted,
            "config": {
                "handle_missing": self.config.handle_missing,
                "imputation_method": self.config.imputation_method,
                "scaling_method": self.config.scaling_method,
                "outlier_method": self.config.outlier_method,
                "feature_selection": self.config.feature_selection,
                "n_features": self.config.n_features,
                "pca_components": self.config.pca_components,
                "time_series_resample": self.config.time_series_resample,
                "resample_freq": self.config.resample_freq
            },
            "fitted_components": {
                "scalers": list(self.scalers.keys()),
                "imputers": list(self.imputers.keys()),
                "feature_selectors": list(self.feature_selectors.keys()),
                "pca": self.pca is not None
            }
        }
    
    def save_preprocessing_artifacts(self, path: str):
        """Save preprocessing artifacts"""
        import joblib
        
        artifacts = {
            "scalers": self.scalers,
            "imputers": self.imputers,
            "feature_selectors": self.feature_selectors,
            "pca": self.pca,
            "config": self.config,
            "is_fitted": self.is_fitted
        }
        
        joblib.dump(artifacts, path)
        logger.info(f"Preprocessing artifacts saved to {path}")
    
    def load_preprocessing_artifacts(self, path: str):
        """Load preprocessing artifacts"""
        import joblib
        
        artifacts = joblib.load(path)
        
        self.scalers = artifacts["scalers"]
        self.imputers = artifacts["imputers"]
        self.feature_selectors = artifacts["feature_selectors"]
        self.pca = artifacts["pca"]
        self.is_fitted = artifacts["is_fitted"]
        
        logger.info(f"Preprocessing artifacts loaded from {path}")
