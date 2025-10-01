"""
Weather Data Ingestion Pipeline
Handles weather data from OpenWeatherMap API for energy demand correlation
"""

import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from kafka import KafkaProducer
import boto3
import yaml
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WeatherData:
    """Data class for weather observations"""
    city: str
    timestamp: datetime
    temperature_celsius: float
    humidity_percent: float
    pressure_hpa: float
    wind_speed_ms: float
    wind_direction_degrees: float
    cloud_cover_percent: float
    visibility_km: float
    uv_index: Optional[float] = None
    precipitation_mm: Optional[float] = None

class WeatherDataIngestionPipeline:
    """Main class for weather data ingestion"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the weather data ingestion pipeline"""
        self.config = self._load_config(config_path)
        self.kafka_producer = self._init_kafka_producer()
        self.s3_client = boto3.client('s3')
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _init_kafka_producer(self) -> KafkaProducer:
        """Initialize Kafka producer for weather data streaming"""
        kafka_config = self.config['kafka']
        return KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            security_protocol=kafka_config.get('security_protocol', 'PLAINTEXT')
        )
    
    def fetch_weather_data(self, city: str) -> Optional[WeatherData]:
        """Fetch weather data for a specific city"""
        try:
            weather_config = self.config['data_sources']['weather']
            
            params = {
                'q': city,
                'appid': weather_config['api_key'],
                'units': 'metric'
            }
            
            response = requests.get(
                weather_config['api_endpoint'],
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            return self._parse_weather_data(data, city)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather data for {city}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing weather data for {city}: {e}")
            return None
    
    def _parse_weather_data(self, data: Dict[str, Any], city: str) -> WeatherData:
        """Parse OpenWeatherMap API response"""
        main = data['main']
        wind = data.get('wind', {})
        clouds = data.get('clouds', {})
        weather = data.get('weather', [{}])[0]
        
        return WeatherData(
            city=city,
            timestamp=datetime.fromtimestamp(data['dt']),
            temperature_celsius=main['temp'],
            humidity_percent=main['humidity'],
            pressure_hpa=main['pressure'],
            wind_speed_ms=wind.get('speed', 0.0),
            wind_direction_degrees=wind.get('deg', 0.0),
            cloud_cover_percent=clouds.get('all', 0.0),
            visibility_km=data.get('visibility', 0) / 1000.0,
            uv_index=None,  # Not available in basic API
            precipitation_mm=None  # Not available in basic API
        )
    
    def validate_weather_data(self, weather: WeatherData) -> bool:
        """Validate weather data for reasonable values"""
        # Check temperature range (reasonable for Germany)
        if not (-30 <= weather.temperature_celsius <= 50):
            logger.warning(f"Invalid temperature: {weather.temperature_celsius}")
            return False
        
        # Check humidity range
        if not (0 <= weather.humidity_percent <= 100):
            logger.warning(f"Invalid humidity: {weather.humidity_percent}")
            return False
        
        # Check pressure range
        if not (950 <= weather.pressure_hpa <= 1050):
            logger.warning(f"Invalid pressure: {weather.pressure_hpa}")
            return False
        
        # Check wind speed range
        if not (0 <= weather.wind_speed_ms <= 50):
            logger.warning(f"Invalid wind speed: {weather.wind_speed_ms}")
            return False
        
        # Check wind direction range
        if not (0 <= weather.wind_direction_degrees <= 360):
            logger.warning(f"Invalid wind direction: {weather.wind_direction_degrees}")
            return False
        
        return True
    
    def calculate_energy_demand_factor(self, weather: WeatherData) -> float:
        """Calculate energy demand factor based on weather conditions"""
        # Base factor
        factor = 1.0
        
        # Temperature effect (heating/cooling demand)
        temp = weather.temperature_celsius
        if temp < 15:  # Heating demand
            factor += (15 - temp) * 0.02
        elif temp > 25:  # Cooling demand
            factor += (temp - 25) * 0.03
        
        # Humidity effect
        if weather.humidity_percent > 80:
            factor += 0.1  # Higher humidity increases energy demand
        
        # Wind effect (wind chill/heat index)
        if weather.wind_speed_ms > 10:
            if temp < 10:
                factor += 0.05  # Wind chill increases heating demand
            elif temp > 25:
                factor += 0.03  # Wind can reduce cooling demand
        
        # Cloud cover effect (affects solar generation)
        if weather.cloud_cover_percent > 70:
            factor += 0.05  # Less solar generation, more grid demand
        
        return max(0.5, min(2.0, factor))  # Clamp between 0.5 and 2.0
    
    def process_all_cities(self):
        """Process weather data for all configured cities"""
        cities = self.config['data_sources']['weather']['cities']
        
        for city in cities:
            logger.info(f"Processing weather data for {city}")
            
            weather = self.fetch_weather_data(city)
            if weather and self.validate_weather_data(weather):
                # Calculate energy demand factor
                weather.energy_demand_factor = self.calculate_energy_demand_factor(weather)
                
                # Store in S3
                self._store_weather_data_s3(weather)
                
                # Send to Kafka
                self._send_to_kafka(weather)
                
                logger.info(f"Successfully processed weather data for {city}")
            else:
                logger.warning(f"Failed to process weather data for {city}")
    
    def _store_weather_data_s3(self, weather: WeatherData):
        """Store weather data to S3"""
        s3_key = f"raw/weather_data/{weather.timestamp.strftime('%Y/%m/%d')}/{weather.city}_{weather.timestamp.isoformat()}.json"
        
        weather_dict = {
            "city": weather.city,
            "timestamp": weather.timestamp.isoformat(),
            "temperature_celsius": weather.temperature_celsius,
            "humidity_percent": weather.humidity_percent,
            "pressure_hpa": weather.pressure_hpa,
            "wind_speed_ms": weather.wind_speed_ms,
            "wind_direction_degrees": weather.wind_direction_degrees,
            "cloud_cover_percent": weather.cloud_cover_percent,
            "visibility_km": weather.visibility_km,
            "uv_index": weather.uv_index,
            "precipitation_mm": weather.precipitation_mm,
            "energy_demand_factor": getattr(weather, 'energy_demand_factor', 1.0),
            "processing_timestamp": datetime.utcnow().isoformat()
        }
        
        self.s3_client.put_object(
            Bucket=self.config['aws']['s3_bucket'],
            Key=s3_key,
            Body=json.dumps(weather_dict),
            ContentType='application/json'
        )
    
    def _send_to_kafka(self, weather: WeatherData):
        """Send weather data to Kafka topic"""
        kafka_data = {
            "city": weather.city,
            "timestamp": weather.timestamp.isoformat(),
            "temperature_celsius": weather.temperature_celsius,
            "humidity_percent": weather.humidity_percent,
            "pressure_hpa": weather.pressure_hpa,
            "wind_speed_ms": weather.wind_speed_ms,
            "wind_direction_degrees": weather.wind_direction_degrees,
            "cloud_cover_percent": weather.cloud_cover_percent,
            "visibility_km": weather.visibility_km,
            "uv_index": weather.uv_index,
            "precipitation_mm": weather.precipitation_mm,
            "energy_demand_factor": getattr(weather, 'energy_demand_factor', 1.0)
        }
        
        self.kafka_producer.send(
            'weather-data',
            value=kafka_data
        )
    
    def run_continuous_ingestion(self, interval_seconds: int = 1800):  # 30 minutes
        """Run continuous weather data ingestion"""
        logger.info(f"Starting continuous weather data ingestion (interval: {interval_seconds}s)")
        
        try:
            while True:
                self.process_all_cities()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("Stopping continuous weather data ingestion")

def main():
    """Main function to run the weather data ingestion"""
    pipeline = WeatherDataIngestionPipeline()
    
    # Run continuous ingestion
    pipeline.run_continuous_ingestion()

if __name__ == "__main__":
    main()
