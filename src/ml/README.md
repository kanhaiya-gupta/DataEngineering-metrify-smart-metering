# 🧠 ML Module - Metrify Smart Metering Data Pipeline

> **Comprehensive Machine Learning capabilities for smart meter data processing, forecasting, and optimization**

## 📋 Overview

This module provides a complete ML pipeline for the Metrify Smart Metering Data Pipeline, including:

- **Feature Engineering**: Automated feature extraction and preprocessing
- **Model Training**: Multiple ML models for different use cases
- **Model Serving**: Production-ready model serving with A/B testing
- **Monitoring**: Comprehensive model and data drift monitoring
- **Feature Store**: Centralized feature management and serving

## 🏗️ Architecture

```
src/ml/
├── pipelines/           # ML Pipeline Components
│   ├── feature_engineering.py    # Automated feature engineering
│   ├── data_preprocessing.py     # Data cleaning and preprocessing
│   ├── model_training.py         # Model training pipeline
│   └── model_inference.py        # Real-time inference
├── models/              # Specialized ML Models
│   ├── consumption_forecasting.py    # Energy consumption prediction
│   ├── anomaly_detection.py          # Anomaly detection models
│   └── grid_optimization.py          # Grid optimization models
├── features/            # Feature Store
│   ├── feature_store.py         # Feature storage and serving
│   ├── feature_validator.py     # Feature validation
│   └── feature_serving.py       # Real-time feature serving
├── monitoring/          # ML Monitoring
│   ├── model_monitor.py         # Model performance monitoring
│   ├── drift_detector.py        # Data and model drift detection
│   └── performance_tracker.py   # Performance tracking
├── serving/             # Model Serving
│   ├── model_server.py          # FastAPI model server
│   ├── inference_api.py         # Inference API endpoints
│   ├── ab_testing.py            # A/B testing framework
│   └── model_registry.py        # Model versioning and registry
└── mlflow_setup.py      # MLflow configuration and utilities
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure ML Pipeline

```python
from src.ml.mlflow_setup import get_mlflow_manager
from src.ml.pipelines.feature_engineering import FeatureEngineeringPipeline, FeatureConfig
from src.ml.models.consumption_forecasting import ConsumptionForecastingModel, ConsumptionModelConfig

# Setup MLflow
mlflow_manager = get_mlflow_manager()

# Configure feature engineering
feature_config = FeatureConfig(
    lookback_window=24,
    forecast_horizon=1,
    include_weather=True,
    include_grid_status=True
)

# Configure model training
model_config = ConsumptionModelConfig(
    model_type="lstm",
    sequence_length=24,
    forecast_horizon=1,
    lstm_units=[64, 32],
    dropout_rate=0.2
)
```

### 3. Train Models

```bash
# Train all models
python scripts/train_ml_models.py --config config/ml_config.yaml

# Train specific models
python scripts/train_ml_models.py --models consumption anomaly --data-size 50000
```

### 4. Start Model Server

```python
from src.ml.serving.model_server import ModelServer, ServerConfig

# Configure server
config = ServerConfig(
    host="0.0.0.0",
    port=8000,
    model_path="models/consumption_forecasting_model.h5"
)

# Start server
server = ModelServer(config)
server.start_server()
```

## 📊 Supported Models

### 1. Consumption Forecasting
- **LSTM**: Long Short-Term Memory networks
- **GRU**: Gated Recurrent Units
- **CNN-LSTM**: Convolutional + LSTM hybrid
- **Transformer**: Attention-based models
- **Attention-LSTM**: LSTM with attention mechanism

### 2. Anomaly Detection
- **Autoencoder**: Reconstruction-based anomaly detection
- **LSTM-Autoencoder**: Time series autoencoder
- **Isolation Forest**: Unsupervised anomaly detection
- **One-Class SVM**: Novelty detection

### 3. Grid Optimization
- **Multi-Output Models**: Multiple optimization targets
- **Ensemble Models**: Combined optimization strategies
- **Reinforcement Learning**: RL-based optimization

## 🔧 Configuration

### ML Configuration (`config/ml_config.yaml`)

```yaml
# Model Training
training:
  consumption_forecasting:
    model_type: "lstm"
    sequence_length: 24
    forecast_horizon: 1
    lstm_units: [64, 32]
    dropout_rate: 0.2

# Feature Engineering
feature_engineering:
  lookback_window: 24
  include_weather: true
  include_grid_status: true
  include_time_features: true

# Model Serving
serving:
  host: "0.0.0.0"
  port: 8000
  max_batch_size: 32
  enable_metrics: true

# MLflow
mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "smart_meter_ml"
```

## 📈 Monitoring & Observability

### Model Performance Monitoring
- Real-time performance metrics
- Model drift detection
- Data quality monitoring
- Automated alerting

### MLflow Integration
- Experiment tracking
- Model versioning
- Model registry
- Artifact storage

### A/B Testing
- Traffic splitting
- Statistical significance testing
- Automated model promotion
- Performance comparison

## 🎯 Use Cases

### 1. Energy Consumption Forecasting
```python
# Train consumption forecasting model
model = ConsumptionForecastingModel(model_config)
results = model.train(X_train, y_train, X_val, y_val)

# Make predictions
predictions = model.predict(X_test)
```

### 2. Anomaly Detection
```python
# Train anomaly detection model
model = AnomalyDetectionModel(anomaly_config)
results = model.train(X_train)

# Detect anomalies
anomalies = model.predict(X_test)
```

### 3. Grid Optimization
```python
# Train grid optimization model
model = GridOptimizationModel(grid_config)
results = model.train(X_train, y_train, X_val, y_val)

# Get optimization recommendations
recommendations = model.optimize_grid(X_test)
```

## 🔄 Feature Store

### Register Features
```python
from src.ml.features.feature_store import FeatureStore, FeatureStoreConfig

# Setup feature store
config = FeatureStoreConfig()
feature_store = FeatureStore(config)

# Register feature
feature_id = feature_store.register_feature(
    name="consumption_lag_24h",
    entity_type="smart_meter",
    feature_type="numeric",
    description="24-hour lagged consumption"
)
```

### Store Feature Values
```python
# Store feature values
feature_store.store_feature_values(
    feature_id=feature_id,
    entity_id="meter_001",
    value=45.6,
    timestamp=datetime.now()
)
```

### Retrieve Features
```python
# Get latest feature values
values = feature_store.get_latest_feature_values(
    feature_names=["consumption_lag_24h", "temperature"],
    entity_id="meter_001"
)
```

## 🧪 A/B Testing

### Create A/B Test
```python
from src.ml.serving.ab_testing import ABTestConfig, get_ab_test_manager

# Configure A/B test
config = ABTestConfig(
    test_name="consumption_model_v2",
    control_model="consumption_v1",
    treatment_model="consumption_v2",
    traffic_split=0.1,
    success_metric="f1_score",
    success_threshold=0.05
)

# Create test
ab_manager = get_ab_test_manager()
test_id = ab_manager.create_test(config)
```

### Route Requests
```python
# Start test
test = ab_manager.get_test(test_id)
test.start_test()

# Route requests
variant = test.route_request(request_id="req_001", user_id="user_123")
```

## 📊 Monitoring Dashboard

### Model Performance
- Accuracy, Precision, Recall, F1-Score
- Response time and throughput
- Error rates and success rates

### Data Drift Detection
- Feature distribution changes
- Statistical significance testing
- Automated drift alerts

### A/B Test Results
- Statistical significance
- Performance comparison
- Recommendation engine

## 🚀 Production Deployment

### Docker Deployment
```dockerfile
FROM tensorflow/tensorflow:2.15.0
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ /app/src/
WORKDIR /app
CMD ["python", "src/ml/serving/model_server.py"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model-server
  template:
    spec:
      containers:
      - name: model-server
        image: metrify/ml-server:latest
        ports:
        - containerPort: 8000
```

## 📚 API Documentation

### Model Serving API

#### Health Check
```http
GET /health
```

#### Make Prediction
```http
POST /predict
Content-Type: application/json

{
  "data": [[1.0, 2.0, 3.0, ...]],
  "model_name": "consumption_forecasting"
}
```

#### Batch Prediction
```http
POST /predict/batch
Content-Type: application/json

{
  "data": [
    [[1.0, 2.0, 3.0, ...]],
    [[4.0, 5.0, 6.0, ...]]
  ]
}
```

#### Get Model Info
```http
GET /model/info
```

#### Get Metrics
```http
GET /metrics
```

## 🔍 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check model file path
   - Verify model format compatibility
   - Check TensorFlow version

2. **Feature Store Connection Issues**
   - Verify database credentials
   - Check Redis connection
   - Validate network connectivity

3. **A/B Test Routing Issues**
   - Ensure test is running
   - Check traffic split configuration
   - Verify user ID consistency

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging for ML components
logger = logging.getLogger('src.ml')
logger.setLevel(logging.DEBUG)
```

## 📈 Performance Optimization

### Model Optimization
- Quantization for faster inference
- Model pruning for smaller size
- Batch processing for efficiency

### Caching Strategy
- Feature value caching
- Model prediction caching
- Redis-based distributed caching

### Scaling
- Horizontal pod autoscaling
- Load balancing
- Model sharding

## 🔐 Security

### Model Security
- Model encryption at rest
- Secure model serving
- Access control and authentication

### Data Privacy
- PII detection and masking
- Data anonymization
- GDPR compliance

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation
- Contact the ML team

---

*This ML module is part of the Metrify Smart Metering Data Pipeline v2*
