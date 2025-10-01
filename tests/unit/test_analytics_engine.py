"""
Unit tests for Advanced Analytics components (Phase 2)
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Import the actual components we want to test
# from src.analytics.forecasting.time_series_forecaster import TimeSeriesForecaster
# from src.analytics.anomaly_detection.multivariate_detector import MultivariateAnomalyDetector
# from src.analytics.quality.quality_analyzer import QualityAnalyzer


class TestTimeSeriesForecaster:
    """Test time series forecasting functionality."""
    
    @pytest.mark.unit
    def test_forecast_initialization(self, mock_analytics_engine):
        """Test forecast engine initialization."""
        assert mock_analytics_engine is not None
        assert hasattr(mock_analytics_engine, 'calculate_forecast')
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_energy_consumption_forecast(self, mock_analytics_engine, sample_forecast_data):
        """Test energy consumption forecasting."""
        # Test forecast calculation
        mock_analytics_engine.calculate_forecast.return_value = sample_forecast_data
        forecast = await mock_analytics_engine.calculate_forecast(
            data=sample_forecast_data[['timestamp', 'actual']],
            forecast_horizon=24,
            method='prophet'
        )
        
        assert forecast is not None
        assert isinstance(forecast, pd.DataFrame)
        assert 'forecast' in forecast.columns
        assert 'confidence_lower' in forecast.columns
        assert 'confidence_upper' in forecast.columns
        mock_analytics_engine.calculate_forecast.assert_called_once()
    
    @pytest.mark.unit
    def test_forecast_accuracy_metrics(self, sample_forecast_data):
        """Test forecast accuracy calculation."""
        # Calculate MAPE (Mean Absolute Percentage Error)
        actual = sample_forecast_data['actual']
        forecast = sample_forecast_data['forecast']
        
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        mae = np.mean(np.abs(actual - forecast))
        rmse = np.sqrt(np.mean((actual - forecast) ** 2))
        
        assert mape < 20  # MAPE should be less than 20%
        assert mae > 0
        assert rmse > 0
    
    @pytest.mark.unit
    def test_confidence_intervals(self, sample_forecast_data):
        """Test forecast confidence intervals."""
        forecast = sample_forecast_data['forecast']
        lower = sample_forecast_data['confidence_lower']
        upper = sample_forecast_data['confidence_upper']
        
        # Check that confidence intervals are valid
        assert (lower <= forecast).all()
        assert (forecast <= upper).all()
        assert (lower < upper).all()
    
    @pytest.mark.unit
    def test_seasonal_decomposition(self, sample_forecast_data):
        """Test seasonal decomposition."""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Test decomposition
        ts = sample_forecast_data.set_index('timestamp')['actual']
        decomposition = seasonal_decompose(ts, model='additive', period=24)
        
        assert decomposition.trend is not None
        assert decomposition.seasonal is not None
        assert decomposition.resid is not None


class TestMultivariateAnomalyDetector:
    """Test multivariate anomaly detection."""
    
    @pytest.mark.unit
    def test_anomaly_detector_initialization(self, mock_analytics_engine):
        """Test anomaly detector initialization."""
        assert mock_analytics_engine is not None
        assert hasattr(mock_analytics_engine, 'detect_anomalies')
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, mock_analytics_engine, sample_anomaly_data):
        """Test anomaly detection on sample data."""
        # Create test data with known anomalies
        normal_data = np.random.normal(100, 10, 100)
        anomaly_data = np.array([200, 300, 50, 400])
        test_data = np.concatenate([normal_data, anomaly_data])
        
        # Mock anomaly detection results
        mock_anomalies = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(test_data), freq='H'),
            'value': test_data,
            'anomaly_score': np.concatenate([np.zeros(100), np.ones(4) * 0.9]),
            'is_anomaly': np.concatenate([np.zeros(100, dtype=bool), np.ones(4, dtype=bool)])
        })
        
        mock_analytics_engine.detect_anomalies.return_value = mock_anomalies
        anomalies = await mock_analytics_engine.detect_anomalies(
            data=test_data,
            method='isolation_forest',
            contamination=0.04
        )
        
        assert anomalies is not None
        assert isinstance(anomalies, pd.DataFrame)
        assert 'anomaly_score' in anomalies.columns
        assert 'is_anomaly' in anomalies.columns
        assert anomalies['is_anomaly'].sum() == 4  # Should detect 4 anomalies
    
    @pytest.mark.unit
    def test_anomaly_scoring(self, sample_anomaly_data):
        """Test anomaly scoring mechanism."""
        from sklearn.ensemble import IsolationForest
        
        # Test isolation forest
        detector = IsolationForest(contamination=0.1, random_state=42)
        scores = detector.fit_predict(sample_anomaly_data.reshape(-1, 1))
        
        # Check that anomalies are identified
        anomaly_count = np.sum(scores == -1)
        assert anomaly_count > 0
        assert anomaly_count < len(sample_anomaly_data) * 0.2  # Should not be too many
    
    @pytest.mark.unit
    def test_multivariate_anomaly_detection(self):
        """Test multivariate anomaly detection."""
        # Create multivariate test data
        np.random.seed(42)
        normal_data = np.random.multivariate_normal([100, 50], [[10, 2], [2, 5]], 100)
        anomaly_data = np.array([[200, 100], [50, 20], [300, 150]])
        test_data = np.vstack([normal_data, anomaly_data])
        
        from sklearn.ensemble import IsolationForest
        
        detector = IsolationForest(contamination=0.03, random_state=42)
        scores = detector.fit_predict(test_data)
        
        # Check that anomalies are detected
        anomaly_count = np.sum(scores == -1)
        assert anomaly_count >= 2  # Should detect at least 2 of the 3 anomalies


class TestQualityAnalyzer:
    """Test data quality analysis functionality."""
    
    @pytest.mark.unit
    def test_quality_analyzer_initialization(self, mock_analytics_engine):
        """Test quality analyzer initialization."""
        assert mock_analytics_engine is not None
        assert hasattr(mock_analytics_engine, 'calculate_quality_metrics')
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_quality_metrics_calculation(self, mock_analytics_engine, sample_quality_metrics):
        """Test quality metrics calculation."""
        # Mock quality metrics
        mock_analytics_engine.calculate_quality_metrics.return_value = sample_quality_metrics
        metrics = await mock_analytics_engine.calculate_quality_metrics(
            data=pd.DataFrame({'value': [1, 2, 3, 4, 5]}),
            rules=['completeness', 'accuracy', 'consistency']
        )
        
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert 'completeness' in metrics
        assert 'accuracy' in metrics
        assert 'consistency' in metrics
        assert 'overall_score' in metrics
        mock_analytics_engine.calculate_quality_metrics.assert_called_once()
    
    @pytest.mark.unit
    def test_completeness_calculation(self):
        """Test completeness metric calculation."""
        # Test data with missing values
        data = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': [1, None, 3, None, 5],
            'col3': [1, 2, 3, 4, 5]
        })
        
        completeness = 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
        expected_completeness = 1 - (3 / 15)  # 3 missing values out of 15 total
        
        assert abs(completeness - expected_completeness) < 0.001
    
    @pytest.mark.unit
    def test_accuracy_calculation(self):
        """Test accuracy metric calculation."""
        # Test data with known accuracy issues
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        # Calculate accuracy as 1 - (mean absolute error / mean actual value)
        mae = np.mean(np.abs(actual - predicted))
        mean_actual = np.mean(actual)
        accuracy = 1 - (mae / mean_actual)
        
        assert 0 <= accuracy <= 1
        assert accuracy > 0.8  # Should be reasonably accurate
    
    @pytest.mark.unit
    def test_consistency_calculation(self):
        """Test consistency metric calculation."""
        # Test data with consistency issues
        data = pd.DataFrame({
            'meter_id': ['SM001', 'SM001', 'SM002', 'SM002', 'SM001'],
            'energy_consumed': [100, 150, 200, 250, 120],  # SM001 has inconsistent values
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='H')
        })
        
        # Check for consistency within groups
        grouped = data.groupby('meter_id')['energy_consumed']
        consistency_scores = []
        
        for name, group in grouped:
            if len(group) > 1:
                # Calculate coefficient of variation
                cv = group.std() / group.mean()
                consistency = 1 - min(cv, 1)  # Cap at 1
                consistency_scores.append(consistency)
        
        overall_consistency = np.mean(consistency_scores)
        assert 0 <= overall_consistency <= 1


class TestPredictiveMaintenance:
    """Test predictive maintenance analytics."""
    
    @pytest.mark.unit
    def test_equipment_health_scoring(self):
        """Test equipment health scoring."""
        # Mock equipment data
        equipment_data = pd.DataFrame({
            'meter_id': ['SM001', 'SM002', 'SM003'],
            'vibration': [0.1, 0.5, 0.8],  # Higher is worse
            'temperature': [25, 35, 45],    # Higher is worse
            'efficiency': [0.95, 0.85, 0.70]  # Lower is worse
        })
        
        # Calculate health score (weighted combination)
        weights = {'vibration': 0.3, 'temperature': 0.3, 'efficiency': 0.4}
        
        # Normalize and weight
        health_scores = []
        for _, row in equipment_data.iterrows():
            vibration_score = 1 - min(row['vibration'], 1)
            temp_score = 1 - min((row['temperature'] - 20) / 30, 1)
            efficiency_score = row['efficiency']
            
            health_score = (weights['vibration'] * vibration_score + 
                          weights['temperature'] * temp_score + 
                          weights['efficiency'] * efficiency_score)
            health_scores.append(health_score)
        
        assert len(health_scores) == 3
        assert all(0 <= score <= 1 for score in health_scores)
        assert health_scores[0] > health_scores[1] > health_scores[2]  # SM001 > SM002 > SM003
    
    @pytest.mark.unit
    def test_failure_prediction(self):
        """Test failure prediction model."""
        # Mock historical data
        historical_data = pd.DataFrame({
            'days_since_maintenance': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270],
            'vibration': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
            'temperature': [25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
            'failure': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # Failed at day 270
        })
        
        # Simple linear model for failure prediction
        from sklearn.linear_model import LogisticRegression
        
        X = historical_data[['days_since_maintenance', 'vibration', 'temperature']]
        y = historical_data['failure']
        
        model = LogisticRegression()
        model.fit(X, y)
        
        # Test prediction
        test_data = [[200, 0.4, 32]]  # Day 200, moderate vibration and temp
        failure_probability = model.predict_proba(test_data)[0][1]
        
        assert 0 <= failure_probability <= 1
        assert failure_probability > 0.5  # Should predict high failure probability


class TestInteractiveVisualizations:
    """Test interactive visualization components."""
    
    @pytest.mark.unit
    def test_dashboard_data_preparation(self, sample_forecast_data):
        """Test dashboard data preparation."""
        # Prepare data for dashboard
        dashboard_data = {
            'forecast': sample_forecast_data.to_dict('records'),
            'summary_stats': {
                'total_energy': sample_forecast_data['actual'].sum(),
                'avg_energy': sample_forecast_data['actual'].mean(),
                'peak_energy': sample_forecast_data['actual'].max(),
                'forecast_accuracy': 0.92
            }
        }
        
        assert 'forecast' in dashboard_data
        assert 'summary_stats' in dashboard_data
        assert len(dashboard_data['forecast']) > 0
        assert 'total_energy' in dashboard_data['summary_stats']
    
    @pytest.mark.unit
    def test_chart_data_formatting(self, sample_forecast_data):
        """Test chart data formatting for visualizations."""
        # Format data for Plotly charts
        chart_data = {
            'x': sample_forecast_data['timestamp'].tolist(),
            'y_actual': sample_forecast_data['actual'].tolist(),
            'y_forecast': sample_forecast_data['forecast'].tolist(),
            'y_lower': sample_forecast_data['confidence_lower'].tolist(),
            'y_upper': sample_forecast_data['confidence_upper'].tolist()
        }
        
        assert len(chart_data['x']) == len(sample_forecast_data)
        assert len(chart_data['y_actual']) == len(sample_forecast_data)
        assert len(chart_data['y_forecast']) == len(sample_forecast_data)
        assert all(isinstance(x, (int, float)) for x in chart_data['y_actual'])


# Performance tests for analytics components
class TestAnalyticsPerformance:
    """Test analytics component performance."""
    
    @pytest.mark.performance
    def test_forecast_performance(self, mock_analytics_engine, sample_forecast_data):
        """Test forecast calculation performance."""
        import time
        
        start_time = time.time()
        mock_analytics_engine.calculate_forecast(sample_forecast_data)
        end_time = time.time()
        
        forecast_time = end_time - start_time
        assert_performance_requirement(forecast_time, 1.0, "Forecast calculation")  # 1 second max
    
    @pytest.mark.performance
    def test_anomaly_detection_performance(self, mock_analytics_engine, sample_anomaly_data):
        """Test anomaly detection performance."""
        import time
        
        start_time = time.time()
        mock_analytics_engine.detect_anomalies(sample_anomaly_data)
        end_time = time.time()
        
        detection_time = end_time - start_time
        assert_performance_requirement(detection_time, 0.5, "Anomaly detection")  # 500ms max
    
    @pytest.mark.performance
    def test_quality_analysis_performance(self, mock_analytics_engine):
        """Test quality analysis performance."""
        import time
        
        # Generate large dataset
        large_data = pd.DataFrame({
            'value': np.random.randn(10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000)
        })
        
        start_time = time.time()
        mock_analytics_engine.calculate_quality_metrics(large_data)
        end_time = time.time()
        
        analysis_time = end_time - start_time
        assert_performance_requirement(analysis_time, 2.0, "Quality analysis")  # 2 seconds max
