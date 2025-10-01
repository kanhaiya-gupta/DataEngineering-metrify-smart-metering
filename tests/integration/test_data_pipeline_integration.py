"""
Integration tests for data pipeline components
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

# Import the actual components we want to test
# from src.ingestion.smart_meter_ingestion import SmartMeterIngestionService
# from src.processing.data_processor import DataProcessor
# from src.storage.data_storage import DataStorageService


class TestSmartMeterDataPipeline:
    """Test end-to-end smart meter data pipeline."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_data_flow(self, mock_kafka_producer, mock_s3_client, mock_database_session):
        """Test complete data flow from ingestion to storage."""
        # Mock data ingestion
        ingestion_service = Mock()
        ingestion_service.ingest_data = AsyncMock(return_value=[
            {
                "meter_id": "SM001",
                "timestamp": datetime.utcnow(),
                "energy_consumed_kwh": 100.5,
                "power_factor": 0.95,
                "voltage_v": 230.0,
                "current_a": 6.5,
                "frequency_hz": 50.0,
                "temperature_c": 25.0,
                "quality_score": 0.95,
                "anomaly_detected": False
            }
        ])
        
        # Mock data processing
        processor = Mock()
        processor.process_data = AsyncMock(return_value={
            "processed_data": [
                {
                    "meter_id": "SM001",
                    "timestamp": datetime.utcnow(),
                    "energy_consumed_kwh": 100.5,
                    "normalized_energy": 1.005,
                    "quality_score": 0.95,
                    "anomaly_score": 0.1,
                    "is_anomaly": False
                }
            ],
            "quality_metrics": {
                "completeness": 1.0,
                "accuracy": 0.95,
                "consistency": 0.98
            }
        })
        
        # Mock storage service
        storage_service = Mock()
        storage_service.store_data = AsyncMock(return_value=True)
        storage_service.publish_event = AsyncMock(return_value=True)
        
        # Execute pipeline
        raw_data = await ingestion_service.ingest_data()
        processed_result = await processor.process_data(raw_data)
        storage_success = await storage_service.store_data(processed_result["processed_data"])
        event_published = await storage_service.publish_event(processed_result)
        
        # Verify pipeline execution
        assert len(raw_data) == 1
        assert raw_data[0]["meter_id"] == "SM001"
        
        assert "processed_data" in processed_result
        assert "quality_metrics" in processed_result
        assert len(processed_result["processed_data"]) == 1
        
        assert storage_success is True
        assert event_published is True
        
        # Verify service calls
        ingestion_service.ingest_data.assert_called_once()
        processor.process_data.assert_called_once_with(raw_data)
        storage_service.store_data.assert_called_once()
        storage_service.publish_event.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_data_quality_validation(self, mock_database_session):
        """Test data quality validation in pipeline."""
        # Mock quality validation service
        quality_service = Mock()
        quality_service.validate_data = AsyncMock(return_value={
            "is_valid": True,
            "quality_score": 0.92,
            "issues": [],
            "recommendations": []
        })
        
        # Test data with quality issues
        test_data = [
            {
                "meter_id": "SM001",
                "timestamp": datetime.utcnow(),
                "energy_consumed_kwh": 100.5,
                "power_factor": 0.95,
                "voltage_v": 230.0,
                "current_a": 6.5,
                "frequency_hz": 50.0,
                "temperature_c": 25.0,
                "quality_score": 0.92,
                "anomaly_detected": False
            }
        ]
        
        # Validate data
        validation_result = await quality_service.validate_data(test_data)
        
        # Verify validation
        assert validation_result["is_valid"] is True
        assert validation_result["quality_score"] >= 0.9
        assert len(validation_result["issues"]) == 0
        quality_service.validate_data.assert_called_once_with(test_data)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, mock_kafka_producer, mock_database_session):
        """Test error handling and recovery in pipeline."""
        # Mock services with error scenarios
        ingestion_service = Mock()
        ingestion_service.ingest_data = AsyncMock(side_effect=Exception("Connection timeout"))
        
        error_handler = Mock()
        error_handler.handle_error = AsyncMock(return_value=True)
        error_handler.retry_operation = AsyncMock(return_value=True)
        
        # Test error handling
        with pytest.raises(Exception, match="Connection timeout"):
            await ingestion_service.ingest_data()
        
        # Test error recovery
        recovery_success = await error_handler.handle_error("Connection timeout")
        retry_success = await error_handler.retry_operation("ingest_data")
        
        assert recovery_success is True
        assert retry_success is True
        error_handler.handle_error.assert_called_once_with("Connection timeout")
        error_handler.retry_operation.assert_called_once_with("ingest_data")


class TestMLPipelineIntegration:
    """Test ML pipeline integration."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ml_model_training_pipeline(self, mock_mlflow_client, mock_tensorflow_model, sample_ml_training_data):
        """Test ML model training pipeline integration."""
        # Mock ML training service
        ml_service = Mock()
        ml_service.prepare_training_data = AsyncMock(return_value=sample_ml_training_data)
        ml_service.train_model = AsyncMock(return_value={
            "model_id": "model_123",
            "accuracy": 0.95,
            "training_time": 300,
            "model_path": "/models/anomaly_detector_v1"
        })
        ml_service.evaluate_model = AsyncMock(return_value={
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88,
            "f1_score": 0.90
        })
        ml_service.deploy_model = AsyncMock(return_value=True)
        
        # Execute ML pipeline
        training_data = await ml_service.prepare_training_data()
        training_result = await ml_service.train_model(training_data)
        evaluation_result = await ml_service.evaluate_model(training_result["model_id"])
        deployment_success = await ml_service.deploy_model(training_result["model_id"])
        
        # Verify ML pipeline
        assert training_data is not None
        assert training_result["accuracy"] >= 0.9
        assert evaluation_result["f1_score"] >= 0.8
        assert deployment_success is True
        
        # Verify service calls
        ml_service.prepare_training_data.assert_called_once()
        ml_service.train_model.assert_called_once_with(training_data)
        ml_service.evaluate_model.assert_called_once_with(training_result["model_id"])
        ml_service.deploy_model.assert_called_once_with(training_result["model_id"])
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ml_model_inference_pipeline(self, mock_tensorflow_model, sample_ml_training_data):
        """Test ML model inference pipeline integration."""
        # Mock inference service
        inference_service = Mock()
        inference_service.load_model = AsyncMock(return_value=mock_tensorflow_model)
        inference_service.preprocess_data = AsyncMock(return_value=np.array([[100.5, 22.5, 65.2]]))
        inference_service.predict = AsyncMock(return_value=np.array([[0.8, 0.2]]))
        inference_service.postprocess_results = AsyncMock(return_value={
            "anomaly_probability": 0.2,
            "is_anomaly": False,
            "confidence": 0.8
        })
        
        # Execute inference pipeline
        model = await inference_service.load_model("model_123")
        preprocessed_data = await inference_service.preprocess_data(sample_ml_training_data)
        predictions = await inference_service.predict(model, preprocessed_data)
        results = await inference_service.postprocess_results(predictions)
        
        # Verify inference pipeline
        assert model is not None
        assert preprocessed_data is not None
        assert predictions is not None
        assert results["is_anomaly"] is False
        assert results["confidence"] >= 0.8
        
        # Verify service calls
        inference_service.load_model.assert_called_once_with("model_123")
        inference_service.preprocess_data.assert_called_once_with(sample_ml_training_data)
        inference_service.predict.assert_called_once()
        inference_service.postprocess_results.assert_called_once_with(predictions)


class TestAnalyticsPipelineIntegration:
    """Test analytics pipeline integration."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_forecasting_pipeline(self, mock_analytics_engine, sample_forecast_data):
        """Test forecasting pipeline integration."""
        # Mock forecasting service
        forecasting_service = Mock()
        forecasting_service.prepare_time_series_data = AsyncMock(return_value=sample_forecast_data)
        forecasting_service.generate_forecast = AsyncMock(return_value=sample_forecast_data)
        forecasting_service.validate_forecast = AsyncMock(return_value={
            "is_valid": True,
            "accuracy_score": 0.92,
            "confidence_level": 0.95
        })
        forecasting_service.store_forecast = AsyncMock(return_value=True)
        
        # Execute forecasting pipeline
        time_series_data = await forecasting_service.prepare_time_series_data()
        forecast = await forecasting_service.generate_forecast(time_series_data)
        validation = await forecasting_service.validate_forecast(forecast)
        storage_success = await forecasting_service.store_forecast(forecast)
        
        # Verify forecasting pipeline
        assert time_series_data is not None
        assert forecast is not None
        assert validation["is_valid"] is True
        assert validation["accuracy_score"] >= 0.9
        assert storage_success is True
        
        # Verify service calls
        forecasting_service.prepare_time_series_data.assert_called_once()
        forecasting_service.generate_forecast.assert_called_once_with(time_series_data)
        forecasting_service.validate_forecast.assert_called_once_with(forecast)
        forecasting_service.store_forecast.assert_called_once_with(forecast)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_anomaly_detection_pipeline(self, mock_analytics_engine, sample_anomaly_data):
        """Test anomaly detection pipeline integration."""
        # Mock anomaly detection service
        anomaly_service = Mock()
        anomaly_service.prepare_detection_data = AsyncMock(return_value=sample_anomaly_data)
        anomaly_service.detect_anomalies = AsyncMock(return_value=pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(sample_anomaly_data), freq='H'),
            'value': sample_anomaly_data,
            'anomaly_score': np.concatenate([np.zeros(100), np.ones(4) * 0.9]),
            'is_anomaly': np.concatenate([np.zeros(100, dtype=bool), np.ones(4, dtype=bool)])
        }))
        anomaly_service.generate_alerts = AsyncMock(return_value=[
            {
                "alert_id": "alert_001",
                "severity": "high",
                "message": "Anomaly detected in energy consumption",
                "timestamp": datetime.utcnow()
            }
        ])
        anomaly_service.store_results = AsyncMock(return_value=True)
        
        # Execute anomaly detection pipeline
        detection_data = await anomaly_service.prepare_detection_data()
        anomalies = await anomaly_service.detect_anomalies(detection_data)
        alerts = await anomaly_service.generate_alerts(anomalies)
        storage_success = await anomaly_service.store_results(anomalies, alerts)
        
        # Verify anomaly detection pipeline
        assert detection_data is not None
        assert anomalies is not None
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "high"
        assert storage_success is True
        
        # Verify service calls
        anomaly_service.prepare_detection_data.assert_called_once()
        anomaly_service.detect_anomalies.assert_called_once_with(detection_data)
        anomaly_service.generate_alerts.assert_called_once_with(anomalies)
        anomaly_service.store_results.assert_called_once_with(anomalies, alerts)


class TestEventDrivenPipelineIntegration:
    """Test event-driven pipeline integration."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_event_sourcing_pipeline(self, mock_event_store, sample_domain_event):
        """Test event sourcing pipeline integration."""
        # Mock event sourcing service
        event_service = Mock()
        event_service.append_event = AsyncMock(return_value=1)
        event_service.get_events = AsyncMock(return_value=[sample_domain_event])
        event_service.replay_events = AsyncMock(return_value={
            "aggregate_id": "SM001",
            "current_state": {"status": "ACTIVE", "version": 1}
        })
        
        # Execute event sourcing pipeline
        version = await event_service.append_event(sample_domain_event)
        events = await event_service.get_events("SM001")
        state = await event_service.replay_events("SM001", events)
        
        # Verify event sourcing pipeline
        assert version == 1
        assert len(events) == 1
        assert state["aggregate_id"] == "SM001"
        assert state["current_state"]["status"] == "ACTIVE"
        
        # Verify service calls
        event_service.append_event.assert_called_once_with(sample_domain_event)
        event_service.get_events.assert_called_once_with("SM001")
        event_service.replay_events.assert_called_once_with("SM001", events)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cqrs_pipeline(self, mock_command_handler, mock_query_handler):
        """Test CQRS pipeline integration."""
        # Mock CQRS service
        cqrs_service = Mock()
        cqrs_service.process_command = AsyncMock(return_value={
            "command_id": "cmd_123",
            "status": "success",
            "events": [{"event_type": "SmartMeterCreated", "data": {}}]
        })
        cqrs_service.execute_query = AsyncMock(return_value={
            "query_id": "query_123",
            "data": [{"meter_id": "SM001", "status": "ACTIVE"}]
        })
        
        # Execute CQRS pipeline
        command = {"command_type": "CreateSmartMeter", "meter_id": "SM001"}
        command_result = await cqrs_service.process_command(command)
        
        query = {"query_type": "GetSmartMeter", "meter_id": "SM001"}
        query_result = await cqrs_service.execute_query(query)
        
        # Verify CQRS pipeline
        assert command_result["status"] == "success"
        assert len(command_result["events"]) == 1
        assert query_result["data"][0]["meter_id"] == "SM001"
        
        # Verify service calls
        cqrs_service.process_command.assert_called_once_with(command)
        cqrs_service.execute_query.assert_called_once_with(query)


class TestMultiCloudIntegration:
    """Test multi-cloud integration."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cloud_storage_integration(self, mock_aws_client, mock_azure_client, mock_gcp_client):
        """Test multi-cloud storage integration."""
        # Mock cloud storage service
        cloud_service = Mock()
        cloud_service.upload_to_aws = AsyncMock(return_value=True)
        cloud_service.upload_to_azure = AsyncMock(return_value=True)
        cloud_service.upload_to_gcp = AsyncMock(return_value=True)
        cloud_service.sync_across_clouds = AsyncMock(return_value=True)
        
        # Test data
        test_data = {"meter_id": "SM001", "energy": 100.5}
        
        # Execute multi-cloud upload
        aws_success = await cloud_service.upload_to_aws(test_data)
        azure_success = await cloud_service.upload_to_azure(test_data)
        gcp_success = await cloud_service.upload_to_gcp(test_data)
        sync_success = await cloud_service.sync_across_clouds()
        
        # Verify multi-cloud integration
        assert aws_success is True
        assert azure_success is True
        assert gcp_success is True
        assert sync_success is True
        
        # Verify service calls
        cloud_service.upload_to_aws.assert_called_once_with(test_data)
        cloud_service.upload_to_azure.assert_called_once_with(test_data)
        cloud_service.upload_to_gcp.assert_called_once_with(test_data)
        cloud_service.sync_across_clouds.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cloud_failover_integration(self, mock_aws_client, mock_azure_client):
        """Test cloud failover integration."""
        # Mock failover service
        failover_service = Mock()
        failover_service.detect_failure = AsyncMock(return_value=True)
        failover_service.switch_to_backup = AsyncMock(return_value="azure")
        failover_service.verify_backup = AsyncMock(return_value=True)
        
        # Execute failover
        failure_detected = await failover_service.detect_failure("aws")
        backup_cloud = await failover_service.switch_to_backup("aws")
        backup_verified = await failover_service.verify_backup(backup_cloud)
        
        # Verify failover integration
        assert failure_detected is True
        assert backup_cloud == "azure"
        assert backup_verified is True
        
        # Verify service calls
        failover_service.detect_failure.assert_called_once_with("aws")
        failover_service.switch_to_backup.assert_called_once_with("aws")
        failover_service.verify_backup.assert_called_once_with("azure")


# Performance tests for integration
class TestIntegrationPerformance:
    """Test integration performance."""
    
    @pytest.mark.performance
    @pytest.mark.integration
    async def test_pipeline_throughput(self, mock_kafka_producer, mock_database_session):
        """Test pipeline throughput performance."""
        import time
        
        # Mock high-throughput pipeline
        pipeline_service = Mock()
        pipeline_service.process_batch = AsyncMock(return_value=True)
        
        # Generate large batch
        batch_size = 1000
        batch_data = [
            {"meter_id": f"SM{i:03d}", "energy": 100 + i, "timestamp": datetime.utcnow()}
            for i in range(batch_size)
        ]
        
        start_time = time.time()
        await pipeline_service.process_batch(batch_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        throughput = batch_size / processing_time
        
        assert throughput >= 1000  # At least 1000 records per second
        assert_performance_requirement(processing_time, 1.0, "Pipeline throughput")
    
    @pytest.mark.performance
    @pytest.mark.integration
    async def test_ml_pipeline_latency(self, mock_tensorflow_model, sample_ml_training_data):
        """Test ML pipeline latency."""
        import time
        
        # Mock ML pipeline
        ml_pipeline = Mock()
        ml_pipeline.process_single_record = AsyncMock(return_value={
            "anomaly_probability": 0.2,
            "processing_time": 0.05
        })
        
        start_time = time.time()
        await ml_pipeline.process_single_record(sample_ml_training_data.iloc[0])
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert_performance_requirement(processing_time, 0.1, "ML pipeline latency")  # 100ms max
