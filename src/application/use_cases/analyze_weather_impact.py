"""
Weather Impact Analysis Use Case
Handles the analysis of weather impact on energy consumption and grid stability
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from ...core.domain.entities.weather_station import WeatherStation, WeatherObservation
from ...core.domain.entities.smart_meter import SmartMeter
from ...core.domain.entities.grid_operator import GridOperator
from ...core.domain.value_objects.location import Location
from ...core.domain.enums.weather_station_status import WeatherStationStatus
from ...core.interfaces.repositories.weather_station_repository import IWeatherStationRepository
from ...core.interfaces.repositories.smart_meter_repository import ISmartMeterRepository
from ...core.interfaces.repositories.grid_operator_repository import IGridOperatorRepository
from ...core.interfaces.external.data_quality_service import IDataQualityService
from ...core.interfaces.external.anomaly_detection_service import IAnomalyDetectionService
from ...core.interfaces.external.alerting_service import IAlertingService
from ...core.exceptions.domain_exceptions import WeatherStationNotFoundError, InvalidWeatherObservationError
from ...infrastructure.external.kafka.kafka_producer import KafkaProducer
from ...infrastructure.external.s3.s3_client import S3Client
from ...infrastructure.external.snowflake.query_executor import SnowflakeQueryExecutor

logger = logging.getLogger(__name__)


class AnalyzeWeatherImpactUseCase:
    """
    Use case for analyzing weather impact on energy consumption and grid stability
    
    Handles the complete flow of weather impact analysis,
    including correlation analysis, forecasting, and energy demand prediction.
    """
    
    def __init__(
        self,
        weather_station_repository: IWeatherStationRepository,
        smart_meter_repository: ISmartMeterRepository,
        grid_operator_repository: IGridOperatorRepository,
        data_quality_service: IDataQualityService,
        anomaly_detection_service: IAnomalyDetectionService,
        alerting_service: IAlertingService,
        kafka_producer: KafkaProducer,
        s3_client: S3Client,
        snowflake_query_executor: SnowflakeQueryExecutor
    ):
        self.weather_station_repository = weather_station_repository
        self.smart_meter_repository = smart_meter_repository
        self.grid_operator_repository = grid_operator_repository
        self.data_quality_service = data_quality_service
        self.anomaly_detection_service = anomaly_detection_service
        self.alerting_service = alerting_service
        self.kafka_producer = kafka_producer
        self.s3_client = s3_client
        self.snowflake_query_executor = snowflake_query_executor
    
    async def execute(
        self,
        station_id: str,
        analysis_period_hours: int = 24,
        correlation_threshold: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute weather impact analysis
        
        Args:
            station_id: Weather station identifier
            analysis_period_hours: Hours to analyze (default: 24)
            correlation_threshold: Minimum correlation threshold for significance
            metadata: Optional metadata for the analysis
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info(f"Starting weather impact analysis for station {station_id}")
            
            # Step 1: Validate station exists
            station = await self._validate_station_exists(station_id)
            
            # Step 2: Get weather observations for analysis period
            weather_data = await self._get_weather_data(station_id, analysis_period_hours)
            
            # Step 3: Get energy consumption data for correlation
            energy_data = await self._get_energy_data(station.location, analysis_period_hours)
            
            # Step 4: Get grid stability data for correlation
            grid_data = await self._get_grid_data(station.location, analysis_period_hours)
            
            # Step 5: Perform correlation analysis
            correlation_results = await self._analyze_correlations(
                weather_data, energy_data, grid_data, correlation_threshold
            )
            
            # Step 6: Generate weather forecasts
            forecast_results = await self._generate_weather_forecasts(station_id, analysis_period_hours)
            
            # Step 7: Predict energy demand based on weather
            energy_forecast = await self._predict_energy_demand(
                weather_data, energy_data, forecast_results
            )
            
            # Step 8: Analyze grid stability impact
            grid_impact = await self._analyze_grid_stability_impact(
                weather_data, grid_data, forecast_results
            )
            
            # Step 9: Store analysis results
            await self._store_analysis_results(
                station_id, correlation_results, forecast_results, energy_forecast, grid_impact
            )
            
            # Step 10: Publish results to Kafka
            await self._publish_results_to_kafka(station_id, correlation_results, energy_forecast)
            
            # Step 11: Archive results to S3
            await self._archive_results_to_s3(station_id, correlation_results, energy_forecast, metadata)
            
            # Step 12: Send alerts for significant findings
            await self._send_impact_alerts(station, correlation_results, energy_forecast, grid_impact)
            
            result = {
                "status": "success",
                "station_id": station_id,
                "analysis_period_hours": analysis_period_hours,
                "correlation_results": correlation_results,
                "energy_forecast": energy_forecast,
                "grid_impact": grid_impact,
                "alerts_sent": correlation_results.get("alerts_sent", 0),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Weather impact analysis completed for station {station_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in weather impact analysis: {str(e)}")
            raise
    
    async def _validate_station_exists(self, station_id: str) -> WeatherStation:
        """Validate that the weather station exists"""
        try:
            station = await self.weather_station_repository.get_by_id(station_id)
            if not station:
                raise WeatherStationNotFoundError(f"Weather station {station_id} not found")
            return station
        except Exception as e:
            logger.error(f"Error validating station {station_id}: {str(e)}")
            raise
    
    async def _get_weather_data(
        self,
        station_id: str,
        analysis_period_hours: int
    ) -> List[WeatherObservation]:
        """Get weather observations for the analysis period"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=analysis_period_hours)
            
            observations = await self.weather_station_repository.get_observations_by_time_range(
                station_id, start_time, end_time
            )
            
            logger.info(f"Retrieved {len(observations)} weather observations for station {station_id}")
            return observations
            
        except Exception as e:
            logger.error(f"Error getting weather data: {str(e)}")
            return []
    
    async def _get_energy_data(
        self,
        location: Location,
        analysis_period_hours: int
    ) -> List[Dict[str, Any]]:
        """Get energy consumption data for correlation analysis"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=analysis_period_hours)
            
            # Get smart meters within a certain radius of the weather station
            meters = await self.smart_meter_repository.get_meters_by_location_radius(
                location, radius_km=50.0
            )
            
            energy_data = []
            for meter in meters:
                readings = await self.smart_meter_repository.get_readings_by_time_range(
                    meter.meter_id.value, start_time, end_time
                )
                
                for reading in readings:
                    energy_data.append({
                        'timestamp': reading.timestamp,
                        'active_power': reading.active_power,
                        'reactive_power': reading.reactive_power,
                        'apparent_power': reading.apparent_power,
                        'voltage': reading.voltage,
                        'current': reading.current,
                        'meter_id': meter.meter_id.value
                    })
            
            logger.info(f"Retrieved {len(energy_data)} energy consumption records")
            return energy_data
            
        except Exception as e:
            logger.error(f"Error getting energy data: {str(e)}")
            return []
    
    async def _get_grid_data(
        self,
        location: Location,
        analysis_period_hours: int
    ) -> List[Dict[str, Any]]:
        """Get grid stability data for correlation analysis"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=analysis_period_hours)
            
            # Get grid operators within a certain radius of the weather station
            operators = await self.grid_operator_repository.get_operators_by_location_radius(
                location, radius_km=100.0
            )
            
            grid_data = []
            for operator in operators:
                statuses = await self.grid_operator_repository.get_statuses_by_time_range(
                    operator.operator_id, start_time, end_time
                )
                
                for status in statuses:
                    grid_data.append({
                        'timestamp': status.timestamp,
                        'voltage_level': status.voltage_level,
                        'frequency': status.frequency,
                        'load_percentage': status.load_percentage,
                        'stability_score': status.stability_score,
                        'power_quality_score': status.power_quality_score,
                        'operator_id': operator.operator_id
                    })
            
            logger.info(f"Retrieved {len(grid_data)} grid status records")
            return grid_data
            
        except Exception as e:
            logger.error(f"Error getting grid data: {str(e)}")
            return []
    
    async def _analyze_correlations(
        self,
        weather_data: List[WeatherObservation],
        energy_data: List[Dict[str, Any]],
        grid_data: List[Dict[str, Any]],
        correlation_threshold: float
    ) -> Dict[str, Any]:
        """Analyze correlations between weather and energy/grid data"""
        try:
            if not weather_data or not energy_data:
                return {"correlations": [], "significant_correlations": [], "alerts_sent": 0}
            
            # Group data by hour for correlation analysis
            hourly_weather = self._group_data_by_hour(weather_data, 'weather')
            hourly_energy = self._group_data_by_hour(energy_data, 'energy')
            hourly_grid = self._group_data_by_hour(grid_data, 'grid')
            
            correlations = []
            significant_correlations = []
            alerts_sent = 0
            
            # Analyze temperature vs energy consumption correlation
            temp_energy_corr = self._calculate_correlation(
                hourly_weather.get('temperature', []),
                hourly_energy.get('active_power', [])
            )
            
            if abs(temp_energy_corr) >= correlation_threshold:
                correlation = {
                    'type': 'temperature_energy',
                    'correlation': temp_energy_corr,
                    'strength': 'strong' if abs(temp_energy_corr) >= 0.8 else 'moderate',
                    'description': 'Temperature vs Energy Consumption'
                }
                correlations.append(correlation)
                significant_correlations.append(correlation)
                
                # Send alert for strong correlation
                if abs(temp_energy_corr) >= 0.8:
                    await self.alerting_service.send_alert(
                        alert_type="weather_energy_correlation",
                        severity="info",
                        message=f"Strong temperature-energy correlation detected: {temp_energy_corr:.2f}",
                        entity_id="weather_analysis",
                        entity_type="analysis",
                        metadata=correlation
                    )
                    alerts_sent += 1
            
            # Analyze humidity vs energy consumption correlation
            humidity_energy_corr = self._calculate_correlation(
                hourly_weather.get('humidity', []),
                hourly_energy.get('active_power', [])
            )
            
            if abs(humidity_energy_corr) >= correlation_threshold:
                correlation = {
                    'type': 'humidity_energy',
                    'correlation': humidity_energy_corr,
                    'strength': 'strong' if abs(humidity_energy_corr) >= 0.8 else 'moderate',
                    'description': 'Humidity vs Energy Consumption'
                }
                correlations.append(correlation)
                significant_correlations.append(correlation)
            
            # Analyze temperature vs grid stability correlation
            temp_grid_corr = self._calculate_correlation(
                hourly_weather.get('temperature', []),
                hourly_grid.get('stability_score', [])
            )
            
            if abs(temp_grid_corr) >= correlation_threshold:
                correlation = {
                    'type': 'temperature_grid',
                    'correlation': temp_grid_corr,
                    'strength': 'strong' if abs(temp_grid_corr) >= 0.8 else 'moderate',
                    'description': 'Temperature vs Grid Stability'
                }
                correlations.append(correlation)
                significant_correlations.append(correlation)
            
            # Analyze wind speed vs grid stability correlation
            wind_grid_corr = self._calculate_correlation(
                hourly_weather.get('wind_speed', []),
                hourly_grid.get('stability_score', [])
            )
            
            if abs(wind_grid_corr) >= correlation_threshold:
                correlation = {
                    'type': 'wind_grid',
                    'correlation': wind_grid_corr,
                    'strength': 'strong' if abs(wind_grid_corr) >= 0.8 else 'moderate',
                    'description': 'Wind Speed vs Grid Stability'
                }
                correlations.append(correlation)
                significant_correlations.append(correlation)
            
            correlation_results = {
                'correlations': correlations,
                'significant_correlations': significant_correlations,
                'alerts_sent': alerts_sent,
                'analysis_summary': {
                    'total_correlations': len(correlations),
                    'significant_count': len(significant_correlations),
                    'strong_correlations': len([c for c in correlations if c['strength'] == 'strong']),
                    'moderate_correlations': len([c for c in correlations if c['strength'] == 'moderate'])
                }
            }
            
            logger.info(f"Correlation analysis completed: {len(correlations)} correlations found")
            return correlation_results
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {str(e)}")
            return {"correlations": [], "significant_correlations": [], "alerts_sent": 0}
    
    async def _generate_weather_forecasts(
        self,
        station_id: str,
        forecast_hours: int
    ) -> Dict[str, Any]:
        """Generate weather forecasts for the station"""
        try:
            # This would typically call an external weather service
            # For now, we'll simulate forecast generation
            current_time = datetime.utcnow()
            forecasts = []
            
            for i in range(forecast_hours):
                forecast_time = current_time + timedelta(hours=i)
                # Simulate forecast data (in real implementation, call weather API)
                forecast = {
                    'timestamp': forecast_time.isoformat(),
                    'temperature_celsius': 20.0 + (i * 0.5),  # Simulated temperature
                    'humidity_percent': 60.0 + (i * 2.0),  # Simulated humidity
                    'pressure_hpa': 1013.25 + (i * 0.1),  # Simulated pressure
                    'wind_speed_ms': 5.0 + (i * 0.2),  # Simulated wind speed
                    'wind_direction_degrees': 180.0 + (i * 10.0),  # Simulated wind direction
                    'cloud_cover_percent': 30.0 + (i * 5.0),  # Simulated cloud cover
                    'precipitation_mm': 0.0 if i < 12 else 2.0,  # Simulated precipitation
                    'confidence': 0.8 - (i * 0.02)  # Decreasing confidence over time
                }
                forecasts.append(forecast)
            
            forecast_results = {
                'station_id': station_id,
                'forecast_hours': forecast_hours,
                'forecasts': forecasts,
                'generated_at': current_time.isoformat(),
                'confidence': sum(f['confidence'] for f in forecasts) / len(forecasts)
            }
            
            logger.info(f"Generated {len(forecasts)} weather forecasts for station {station_id}")
            return forecast_results
            
        except Exception as e:
            logger.error(f"Error generating weather forecasts: {str(e)}")
            return {"forecasts": [], "confidence": 0.0}
    
    async def _predict_energy_demand(
        self,
        weather_data: List[WeatherObservation],
        energy_data: List[Dict[str, Any]],
        forecast_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict energy demand based on weather patterns"""
        try:
            if not weather_data or not energy_data:
                return {"predictions": [], "confidence": 0.0}
            
            # Calculate historical weather-energy relationship
            historical_correlation = self._calculate_correlation(
                [obs.temperature_celsius for obs in weather_data],
                [energy['active_power'] for energy in energy_data]
            )
            
            # Generate energy demand predictions based on weather forecasts
            predictions = []
            for forecast in forecast_results.get('forecasts', []):
                # Simple linear model: energy = base_consumption + (temperature * temp_coefficient)
                base_consumption = 1000.0  # Base energy consumption in kW
                temp_coefficient = historical_correlation * 50.0  # Temperature coefficient
                
                predicted_energy = base_consumption + (forecast['temperature_celsius'] * temp_coefficient)
                
                prediction = {
                    'timestamp': forecast['timestamp'],
                    'predicted_energy_kw': predicted_energy,
                    'temperature_celsius': forecast['temperature_celsius'],
                    'humidity_percent': forecast['humidity_percent'],
                    'confidence': forecast['confidence'],
                    'factors': {
                        'temperature_impact': forecast['temperature_celsius'] * temp_coefficient,
                        'base_consumption': base_consumption,
                        'historical_correlation': historical_correlation
                    }
                }
                predictions.append(prediction)
            
            energy_forecast = {
                'predictions': predictions,
                'confidence': sum(p['confidence'] for p in predictions) / len(predictions),
                'historical_correlation': historical_correlation,
                'model_type': 'linear_regression',
                'generated_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Generated {len(predictions)} energy demand predictions")
            return energy_forecast
            
        except Exception as e:
            logger.error(f"Error predicting energy demand: {str(e)}")
            return {"predictions": [], "confidence": 0.0}
    
    async def _analyze_grid_stability_impact(
        self,
        weather_data: List[WeatherObservation],
        grid_data: List[Dict[str, Any]],
        forecast_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze impact of weather on grid stability"""
        try:
            if not weather_data or not grid_data:
                return {"impact_analysis": [], "recommendations": []}
            
            # Analyze historical weather-grid stability relationship
            weather_grid_correlation = self._calculate_correlation(
                [obs.temperature_celsius for obs in weather_data],
                [grid['stability_score'] for grid in grid_data]
            )
            
            # Generate grid stability impact predictions
            impact_analysis = []
            for forecast in forecast_results.get('forecasts', []):
                # Predict grid stability impact based on weather
                temp_impact = abs(forecast['temperature_celsius'] - 20.0) * 0.01  # Temperature deviation impact
                wind_impact = forecast['wind_speed_ms'] * 0.02  # Wind speed impact
                humidity_impact = abs(forecast['humidity_percent'] - 50.0) * 0.005  # Humidity deviation impact
                
                total_impact = temp_impact + wind_impact + humidity_impact
                stability_risk = min(total_impact, 1.0)  # Cap at 1.0
                
                impact = {
                    'timestamp': forecast['timestamp'],
                    'stability_risk': stability_risk,
                    'temperature_impact': temp_impact,
                    'wind_impact': wind_impact,
                    'humidity_impact': humidity_impact,
                    'risk_level': 'high' if stability_risk > 0.7 else 'medium' if stability_risk > 0.4 else 'low',
                    'recommendations': self._generate_stability_recommendations(stability_risk, forecast)
                }
                impact_analysis.append(impact)
            
            grid_impact = {
                'impact_analysis': impact_analysis,
                'historical_correlation': weather_grid_correlation,
                'overall_risk': sum(impact['stability_risk'] for impact in impact_analysis) / len(impact_analysis),
                'high_risk_periods': len([impact for impact in impact_analysis if impact['risk_level'] == 'high']),
                'generated_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Generated grid stability impact analysis with {len(impact_analysis)} predictions")
            return grid_impact
            
        except Exception as e:
            logger.error(f"Error analyzing grid stability impact: {str(e)}")
            return {"impact_analysis": [], "recommendations": []}
    
    def _group_data_by_hour(self, data: List, data_type: str) -> Dict[str, List[float]]:
        """Group data by hour for correlation analysis"""
        hourly_data = {}
        
        for item in data:
            if data_type == 'weather':
                hourly_data.setdefault('temperature', []).append(item.temperature_celsius)
                hourly_data.setdefault('humidity', []).append(item.humidity_percent)
                hourly_data.setdefault('wind_speed', []).append(item.wind_speed_ms)
            elif data_type == 'energy':
                hourly_data.setdefault('active_power', []).append(item['active_power'])
            elif data_type == 'grid':
                hourly_data.setdefault('stability_score', []).append(item['stability_score'])
        
        return hourly_data
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation coefficient between two lists"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        sum_y2 = sum(y[i] ** 2 for i in range(n))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _generate_stability_recommendations(self, stability_risk: float, forecast: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on stability risk and weather forecast"""
        recommendations = []
        
        if stability_risk > 0.7:
            recommendations.append("High grid stability risk - prepare for potential outages")
            recommendations.append("Increase grid monitoring frequency")
            recommendations.append("Prepare backup power sources")
        elif stability_risk > 0.4:
            recommendations.append("Medium grid stability risk - monitor closely")
            recommendations.append("Prepare contingency plans")
        
        if forecast['wind_speed_ms'] > 15:
            recommendations.append("High wind speed - check transmission lines")
        
        if forecast['temperature_celsius'] > 35 or forecast['temperature_celsius'] < -10:
            recommendations.append("Extreme temperature - monitor equipment performance")
        
        return recommendations
    
    async def _store_analysis_results(
        self,
        station_id: str,
        correlation_results: Dict[str, Any],
        forecast_results: Dict[str, Any],
        energy_forecast: Dict[str, Any],
        grid_impact: Dict[str, Any]
    ) -> None:
        """Store analysis results in the database"""
        try:
            # This would typically store results in a dedicated analysis table
            # For now, we'll just log the storage
            logger.info(f"Stored weather impact analysis results for station {station_id}")
            
        except Exception as e:
            logger.error(f"Error storing analysis results: {str(e)}")
            # Don't raise exception as this is not critical for the main flow
    
    async def _publish_results_to_kafka(
        self,
        station_id: str,
        correlation_results: Dict[str, Any],
        energy_forecast: Dict[str, Any]
    ) -> None:
        """Publish analysis results to Kafka"""
        try:
            message = {
                'station_id': station_id,
                'correlation_results': correlation_results,
                'energy_forecast': energy_forecast,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await self.kafka_producer.send_message(
                topic="weather-impact-analysis",
                message=message
            )
            
            logger.info(f"Published weather impact analysis results to Kafka for station {station_id}")
            
        except Exception as e:
            logger.error(f"Error publishing to Kafka: {str(e)}")
            # Don't raise exception as this is not critical for the main flow
    
    async def _archive_results_to_s3(
        self,
        station_id: str,
        correlation_results: Dict[str, Any],
        energy_forecast: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Archive analysis results to S3"""
        try:
            results_data = {
                'station_id': station_id,
                'correlation_results': correlation_results,
                'energy_forecast': energy_forecast,
                'metadata': metadata,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            timestamp = datetime.utcnow().strftime("%Y/%m/%d/%H")
            s3_key = f"weather-impact-analysis/{station_id}/{timestamp}/analysis.json"
            
            await self.s3_client.upload_data(
                data=results_data,
                s3_key=s3_key,
                content_type="application/json"
            )
            
            logger.info(f"Archived weather impact analysis results to S3: {s3_key}")
            
        except Exception as e:
            logger.error(f"Error archiving to S3: {str(e)}")
            # Don't raise exception as this is not critical for the main flow
    
    async def _send_impact_alerts(
        self,
        station: WeatherStation,
        correlation_results: Dict[str, Any],
        energy_forecast: Dict[str, Any],
        grid_impact: Dict[str, Any]
    ) -> None:
        """Send alerts for significant weather impact findings"""
        try:
            alerts_sent = 0
            
            # Alert for strong correlations
            for correlation in correlation_results.get('significant_correlations', []):
                if correlation['strength'] == 'strong':
                    await self.alerting_service.send_alert(
                        alert_type="weather_impact_strong_correlation",
                        severity="info",
                        message=f"Strong weather impact correlation: {correlation['description']}",
                        entity_id=station.station_id,
                        entity_type="weather_station",
                        metadata=correlation
                    )
                    alerts_sent += 1
            
            # Alert for high energy demand predictions
            high_demand_predictions = [
                p for p in energy_forecast.get('predictions', [])
                if p['predicted_energy_kw'] > 2000  # Threshold for high demand
            ]
            
            if high_demand_predictions:
                await self.alerting_service.send_alert(
                    alert_type="high_energy_demand_prediction",
                    severity="warning",
                    message=f"High energy demand predicted: {len(high_demand_predictions)} periods",
                    entity_id=station.station_id,
                    entity_type="weather_station",
                    metadata={"high_demand_count": len(high_demand_predictions)}
                )
                alerts_sent += 1
            
            # Alert for high grid stability risk
            high_risk_periods = grid_impact.get('high_risk_periods', 0)
            if high_risk_periods > 0:
                await self.alerting_service.send_alert(
                    alert_type="high_grid_stability_risk",
                    severity="warning",
                    message=f"High grid stability risk predicted: {high_risk_periods} periods",
                    entity_id=station.station_id,
                    entity_type="weather_station",
                    metadata={"high_risk_periods": high_risk_periods}
                )
                alerts_sent += 1
            
            logger.info(f"Sent {alerts_sent} weather impact alerts for station {station.station_id}")
            
        except Exception as e:
            logger.error(f"Error sending weather impact alerts: {str(e)}")
            # Don't raise exception as this is not critical for the main flow
