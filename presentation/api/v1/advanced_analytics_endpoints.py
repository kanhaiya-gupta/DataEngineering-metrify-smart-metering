"""
Advanced Analytics API Endpoints
REST API endpoints for advanced analytics including forecasting, anomaly detection, and predictive maintenance
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Path, status, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import analytics components
from src.analytics.forecasting.consumption_forecaster import ConsumptionForecaster
from src.analytics.forecasting.grid_load_predictor import GridLoadPredictor
from src.analytics.forecasting.weather_impact_analyzer import WeatherImpactAnalyzer
from src.analytics.anomaly_detection.multivariate_anomaly_detector import MultivariateAnomalyDetector
from src.analytics.anomaly_detection.real_time_anomaly_scorer import RealTimeAnomalyScorer
from src.analytics.anomaly_detection.anomaly_explainer import AnomalyExplainer
from src.analytics.predictive_maintenance.equipment_failure_predictor import EquipmentFailurePredictor
from src.analytics.predictive_maintenance.maintenance_optimizer import MaintenanceOptimizer, MaintenanceTask, MaintenanceResource
from src.analytics.predictive_maintenance.risk_assessor import RiskAssessor, RiskAssessment, RiskFactor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/advanced-analytics", tags=["Advanced Analytics"])

# Global instances
consumption_forecaster = None
grid_load_predictor = None
weather_impact_analyzer = None
anomaly_detector = None
real_time_scorer = None
anomaly_explainer = None
failure_predictor = None
maintenance_optimizer = None
risk_assessor = None

# Dependency injection
def get_consumption_forecaster():
    """Get consumption forecaster instance"""
    global consumption_forecaster
    if consumption_forecaster is None:
        consumption_forecaster = ConsumptionForecaster()
    return consumption_forecaster

def get_grid_load_predictor():
    """Get grid load predictor instance"""
    global grid_load_predictor
    if grid_load_predictor is None:
        grid_load_predictor = GridLoadPredictor()
    return grid_load_predictor

def get_weather_impact_analyzer():
    """Get weather impact analyzer instance"""
    global weather_impact_analyzer
    if weather_impact_analyzer is None:
        weather_impact_analyzer = WeatherImpactAnalyzer()
    return weather_impact_analyzer

def get_anomaly_detector():
    """Get anomaly detector instance"""
    global anomaly_detector
    if anomaly_detector is None:
        anomaly_detector = MultivariateAnomalyDetector()
    return anomaly_detector

def get_real_time_scorer():
    """Get real-time anomaly scorer instance"""
    global real_time_scorer
    if real_time_scorer is None:
        real_time_scorer = RealTimeAnomalyScorer()
    return real_time_scorer

def get_anomaly_explainer():
    """Get anomaly explainer instance"""
    global anomaly_explainer
    if anomaly_explainer is None:
        anomaly_explainer = AnomalyExplainer()
    return anomaly_explainer

def get_failure_predictor():
    """Get equipment failure predictor instance"""
    global failure_predictor
    if failure_predictor is None:
        failure_predictor = EquipmentFailurePredictor()
    return failure_predictor

def get_maintenance_optimizer():
    """Get maintenance optimizer instance"""
    global maintenance_optimizer
    if maintenance_optimizer is None:
        maintenance_optimizer = MaintenanceOptimizer()
    return maintenance_optimizer

def get_risk_assessor():
    """Get risk assessor instance"""
    global risk_assessor
    if risk_assessor is None:
        risk_assessor = RiskAssessor()
    return risk_assessor

# Request/Response Models
class ForecastingRequest(BaseModel):
    """Request model for forecasting"""
    data: List[Dict[str, Any]] = Field(..., description="Historical data for forecasting")
    horizon: int = Field(24, description="Forecast horizon in hours")
    model_type: str = Field("ensemble", description="Model type to use")

class ForecastingResponse(BaseModel):
    """Response model for forecasting"""
    forecasts: List[Dict[str, Any]]
    model_used: str
    horizon: int
    confidence: float
    generated_at: datetime

class AnomalyDetectionRequest(BaseModel):
    """Request model for anomaly detection"""
    data: List[Dict[str, Any]] = Field(..., description="Data to analyze for anomalies")
    model_name: str = Field("ensemble", description="Anomaly detection model to use")
    threshold: float = Field(0.5, description="Anomaly threshold")

class AnomalyDetectionResponse(BaseModel):
    """Response model for anomaly detection"""
    anomalies: List[Dict[str, Any]]
    total_anomalies: int
    anomaly_rate: float
    model_used: str
    generated_at: datetime

class RiskAssessmentRequest(BaseModel):
    """Request model for risk assessment"""
    equipment_id: str = Field(..., description="Equipment ID to assess")
    data: List[Dict[str, Any]] = Field(..., description="Equipment data for assessment")

class RiskAssessmentResponse(BaseModel):
    """Response model for risk assessment"""
    assessment_id: str
    equipment_id: str
    overall_risk_score: float
    risk_level: str
    risk_factors: List[Dict[str, Any]]
    recommendations: List[str]
    generated_at: datetime

# Forecasting Endpoints
@router.post("/forecasting/consumption", response_model=ForecastingResponse)
async def forecast_consumption(
    request: ForecastingRequest,
    forecaster: ConsumptionForecaster = Depends(get_consumption_forecaster)
):
    """Forecast energy consumption"""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Make forecast
        forecast_result = forecaster.forecast(
            data=df,
            horizon=request.horizon,
            model_type=request.model_type
        )
        
        return ForecastingResponse(
            forecasts=forecast_result.get("forecasts", []),
            model_used=request.model_type,
            horizon=request.horizon,
            confidence=forecast_result.get("confidence", 0.0),
            generated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Consumption forecasting failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Consumption forecasting failed: {str(e)}"
        )

@router.post("/forecasting/grid-load", response_model=ForecastingResponse)
async def forecast_grid_load(
    request: ForecastingRequest,
    predictor: GridLoadPredictor = Depends(get_grid_load_predictor)
):
    """Forecast grid load"""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Make forecast
        forecast_result = predictor.forecast(
            model_name=request.model_type,
            data=df,
            horizon=request.horizon
        )
        
        return ForecastingResponse(
            forecasts=[{"timestamp": datetime.now().isoformat(), "forecast": forecast_result.get("forecast", [])}],
            model_used=request.model_type,
            horizon=request.horizon,
            confidence=0.85,  # Placeholder
            generated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Grid load forecasting failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grid load forecasting failed: {str(e)}"
        )

@router.post("/forecasting/weather-impact")
async def analyze_weather_impact(
    data: List[Dict[str, Any]],
    analyzer: WeatherImpactAnalyzer = Depends(get_weather_impact_analyzer)
):
    """Analyze weather impact on energy consumption"""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Analyze weather impact
        analysis_result = analyzer.analyze_weather_impact(df)
        
        return {
            "weather_impact_analysis": analysis_result,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Weather impact analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Weather impact analysis failed: {str(e)}"
        )

# Anomaly Detection Endpoints
@router.post("/anomaly-detection/detect", response_model=AnomalyDetectionResponse)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    detector: MultivariateAnomalyDetector = Depends(get_anomaly_detector)
):
    """Detect anomalies in data"""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Detect anomalies
        detection_result = detector.detect_anomalies(
            data=df,
            model_name=request.model_name
        )
        
        return AnomalyDetectionResponse(
            anomalies=detection_result.get("anomaly_details", []),
            total_anomalies=detection_result.get("anomaly_count", 0),
            anomaly_rate=detection_result.get("anomaly_count", 0) / len(request.data) if request.data else 0,
            model_used=request.model_name,
            generated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection failed: {str(e)}"
        )

@router.post("/anomaly-detection/real-time")
async def real_time_anomaly_scoring(
    data: Dict[str, Any],
    scorer: RealTimeAnomalyScorer = Depends(get_real_time_scorer)
):
    """Real-time anomaly scoring"""
    try:
        # Add data point and get score
        result = scorer.add_data_point(data)
        
        return {
            "anomaly_score": result.get("anomaly_score", 0.0),
            "is_anomaly": result.get("is_anomaly", False),
            "confidence": result.get("confidence", 0.0),
            "timestamp": result.get("timestamp")
        }
        
    except Exception as e:
        logger.error(f"Real-time anomaly scoring failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Real-time anomaly scoring failed: {str(e)}"
        )

@router.post("/anomaly-detection/explain")
async def explain_anomaly(
    anomaly_data: Dict[str, Any],
    reference_data: List[Dict[str, Any]],
    explainer: AnomalyExplainer = Depends(get_anomaly_explainer)
):
    """Explain why a data point is anomalous"""
    try:
        # Convert reference data to DataFrame
        ref_df = pd.DataFrame(reference_data)
        
        # Get feature names
        feature_names = list(anomaly_data.get("features", {}).keys())
        
        # Explain anomaly
        explanation = explainer.explain_anomaly(
            anomaly_data=anomaly_data,
            reference_data=ref_df,
            feature_names=feature_names
        )
        
        return explanation
        
    except Exception as e:
        logger.error(f"Anomaly explanation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly explanation failed: {str(e)}"
        )

# Predictive Maintenance Endpoints
@router.post("/predictive-maintenance/failure-prediction")
async def predict_equipment_failure(
    data: List[Dict[str, Any]],
    predictor: EquipmentFailurePredictor = Depends(get_failure_predictor)
):
    """Predict equipment failure risk"""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Predict failure risk
        prediction_result = predictor.predict_failure_risk(
            data=df,
            model_name="random_forest"
        )
        
        return {
            "failure_predictions": prediction_result.get("failure_predictions", []),
            "risk_levels": prediction_result.get("risk_levels", []),
            "high_risk_count": prediction_result.get("high_risk_count", 0),
            "critical_risk_count": prediction_result.get("critical_risk_count", 0),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failure prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failure prediction failed: {str(e)}"
        )

@router.post("/predictive-maintenance/optimize-schedule")
async def optimize_maintenance_schedule(
    tasks: List[Dict[str, Any]],
    resources: List[Dict[str, Any]],
    optimizer: MaintenanceOptimizer = Depends(get_maintenance_optimizer)
):
    """Optimize maintenance schedule"""
    try:
        # Clear existing schedule
        optimizer.clear_schedule()
        
        # Add tasks
        for task_data in tasks:
            task = MaintenanceTask(**task_data)
            optimizer.add_task(task)
        
        # Add resources
        for resource_data in resources:
            resource = MaintenanceResource(**resource_data)
            optimizer.add_resource(resource)
        
        # Optimize schedule
        optimization_result = optimizer.optimize_schedule(
            optimization_horizon_days=30,
            objective="balanced"
        )
        
        return {
            "optimization_result": optimization_result,
            "schedule_summary": optimizer.get_schedule_summary(),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Maintenance optimization failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Maintenance optimization failed: {str(e)}"
        )

@router.post("/predictive-maintenance/risk-assessment", response_model=RiskAssessmentResponse)
async def assess_equipment_risk(
    request: RiskAssessmentRequest,
    assessor: RiskAssessor = Depends(get_risk_assessor)
):
    """Assess equipment risk"""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Assess risk
        assessment = assessor.assess_equipment_risk(
            equipment_id=request.equipment_id,
            equipment_data=df
        )
        
        return RiskAssessmentResponse(
            assessment_id=assessment.assessment_id,
            equipment_id=assessment.equipment_id,
            overall_risk_score=assessment.overall_risk_score,
            risk_level=assessment.risk_level.value,
            risk_factors=[{
                "factor_id": f.factor_id,
                "category": f.category.value,
                "name": f.name,
                "probability": f.probability,
                "impact": f.impact,
                "risk_score": f.risk_score
            } for f in assessment.risk_factors],
            recommendations=assessment.recommendations,
            generated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Risk assessment failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk assessment failed: {str(e)}"
        )

# Analytics Status Endpoints
@router.get("/status")
async def get_analytics_status():
    """Get status of all analytics components"""
    try:
        status = {
            "timestamp": datetime.now().isoformat(),
            "components": {
                "consumption_forecaster": consumption_forecaster is not None,
                "grid_load_predictor": grid_load_predictor is not None,
                "weather_impact_analyzer": weather_impact_analyzer is not None,
                "anomaly_detector": anomaly_detector is not None,
                "real_time_scorer": real_time_scorer is not None,
                "anomaly_explainer": anomaly_explainer is not None,
                "failure_predictor": failure_predictor is not None,
                "maintenance_optimizer": maintenance_optimizer is not None,
                "risk_assessor": risk_assessor is not None
            },
            "real_time_scorer_status": real_time_scorer.get_current_status() if real_time_scorer else None
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get analytics status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics status: {str(e)}"
        )

@router.get("/models/train")
async def train_analytics_models(
    background_tasks: BackgroundTasks,
    forecaster: ConsumptionForecaster = Depends(get_consumption_forecaster),
    predictor: GridLoadPredictor = Depends(get_grid_load_predictor),
    analyzer: WeatherImpactAnalyzer = Depends(get_weather_impact_analyzer),
    detector: MultivariateAnomalyDetector = Depends(get_anomaly_detector),
    failure_predictor: EquipmentFailurePredictor = Depends(get_failure_predictor)
):
    """Train all analytics models"""
    try:
        # Generate dummy training data
        def generate_training_data():
            # This would typically load real data from the database
            np.random.seed(42)
            data = {
                'timestamp': pd.date_range(start='2023-01-01', periods=1000, freq='H'),
                'consumption': np.random.normal(100, 20, 1000),
                'temperature': np.random.normal(20, 10, 1000),
                'humidity': np.random.normal(50, 15, 1000),
                'grid_load': np.random.normal(1000, 200, 1000),
                'voltage': np.random.normal(230, 10, 1000),
                'installation_date': '2020-01-01',
                'last_maintenance': '2023-12-01',
                'failure_count': np.random.poisson(2, 1000)
            }
            return pd.DataFrame(data)
        
        training_data = generate_training_data()
        
        # Train models in background
        background_tasks.add_task(
            train_models_background,
            training_data,
            forecaster,
            predictor,
            analyzer,
            detector,
            failure_predictor
        )
        
        return {
            "message": "Model training started in background",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model training failed: {str(e)}"
        )

async def train_models_background(
    data: pd.DataFrame,
    forecaster: ConsumptionForecaster,
    predictor: GridLoadPredictor,
    analyzer: WeatherImpactAnalyzer,
    detector: MultivariateAnomalyDetector,
    failure_predictor: EquipmentFailurePredictor
):
    """Background task to train models"""
    try:
        logger.info("Starting model training...")
        
        # Train consumption forecaster
        forecaster.train_all_models(data)
        logger.info("Consumption forecaster trained")
        
        # Train grid load predictor
        predictor.train_all_models(data)
        logger.info("Grid load predictor trained")
        
        # Train weather impact analyzer
        analyzer.analyze_weather_impact(data)
        logger.info("Weather impact analyzer trained")
        
        # Train anomaly detector
        detector.train_all_models(data)
        logger.info("Anomaly detector trained")
        
        # Train failure predictor
        failure_predictor.train_all_models(data)
        logger.info("Failure predictor trained")
        
        logger.info("All models trained successfully")
        
    except Exception as e:
        logger.error(f"Background model training failed: {str(e)}")

# Health check for advanced analytics
@router.get("/health")
async def health_check():
    """Health check for advanced analytics"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components_loaded": sum([
            consumption_forecaster is not None,
            grid_load_predictor is not None,
            weather_impact_analyzer is not None,
            anomaly_detector is not None,
            real_time_scorer is not None,
            anomaly_explainer is not None,
            failure_predictor is not None,
            maintenance_optimizer is not None,
            risk_assessor is not None
        ]),
        "total_components": 9
    }
