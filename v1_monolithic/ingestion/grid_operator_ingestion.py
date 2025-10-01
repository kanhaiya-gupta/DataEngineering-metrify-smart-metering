"""
Grid Operator Data Ingestion Pipeline
Handles data from various grid operators (TenneT, 50Hertz, etc.)
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
class GridStatus:
    """Data class for grid operator status"""
    operator_name: str
    timestamp: datetime
    total_capacity_mw: float
    available_capacity_mw: float
    load_factor: float
    frequency_hz: float
    voltage_kv: float
    grid_stability_score: float
    renewable_percentage: float
    region: str

class GridOperatorIngestionPipeline:
    """Main class for grid operator data ingestion"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the grid operator ingestion pipeline"""
        self.config = self._load_config(config_path)
        self.kafka_producer = self._init_kafka_producer()
        self.s3_client = boto3.client('s3')
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _init_kafka_producer(self) -> KafkaProducer:
        """Initialize Kafka producer for grid data streaming"""
        kafka_config = self.config['kafka']
        return KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            security_protocol=kafka_config.get('security_protocol', 'PLAINTEXT')
        )
    
    def fetch_grid_status(self, operator_config: Dict[str, Any]) -> Optional[GridStatus]:
        """Fetch grid status from a specific operator"""
        try:
            headers = {
                'Authorization': f"Bearer {operator_config['api_key']}",
                'Content-Type': 'application/json'
            }
            
            response = requests.get(
                operator_config['api_endpoint'],
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Parse operator-specific data format
            if operator_config['name'] == 'TenneT':
                return self._parse_tennet_data(data, operator_config['name'])
            elif operator_config['name'] == '50Hertz':
                return self._parse_50hertz_data(data, operator_config['name'])
            else:
                return self._parse_generic_data(data, operator_config['name'])
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from {operator_config['name']}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing {operator_config['name']} data: {e}")
            return None
    
    def _parse_tennet_data(self, data: Dict[str, Any], operator_name: str) -> GridStatus:
        """Parse TenneT-specific data format"""
        return GridStatus(
            operator_name=operator_name,
            timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
            total_capacity_mw=data['total_capacity'],
            available_capacity_mw=data['available_capacity'],
            load_factor=data['load_factor'],
            frequency_hz=data['frequency'],
            voltage_kv=data['voltage'],
            grid_stability_score=data.get('stability_score', 0.95),
            renewable_percentage=data.get('renewable_percentage', 0.0),
            region=data.get('region', 'Netherlands')
        )
    
    def _parse_50hertz_data(self, data: Dict[str, Any], operator_name: str) -> GridStatus:
        """Parse 50Hertz-specific data format"""
        return GridStatus(
            operator_name=operator_name,
            timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
            total_capacity_mw=data['total_capacity'],
            available_capacity_mw=data['available_capacity'],
            load_factor=data['load_factor'],
            frequency_hz=data['frequency'],
            voltage_kv=data['voltage'],
            grid_stability_score=data.get('stability_score', 0.95),
            renewable_percentage=data.get('renewable_percentage', 0.0),
            region=data.get('region', 'Germany')
        )
    
    def _parse_generic_data(self, data: Dict[str, Any], operator_name: str) -> GridStatus:
        """Parse generic grid operator data format"""
        return GridStatus(
            operator_name=operator_name,
            timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
            total_capacity_mw=data.get('total_capacity', 0.0),
            available_capacity_mw=data.get('available_capacity', 0.0),
            load_factor=data.get('load_factor', 0.0),
            frequency_hz=data.get('frequency', 50.0),
            voltage_kv=data.get('voltage', 400.0),
            grid_stability_score=data.get('stability_score', 0.95),
            renewable_percentage=data.get('renewable_percentage', 0.0),
            region=data.get('region', 'Unknown')
        )
    
    def validate_grid_status(self, status: GridStatus) -> bool:
        """Validate grid status data"""
        # Check frequency is within acceptable range
        if not (49.5 <= status.frequency_hz <= 50.5):
            logger.warning(f"Invalid frequency: {status.frequency_hz}")
            return False
        
        # Check voltage is reasonable
        if not (380 <= status.voltage_kv <= 420):
            logger.warning(f"Invalid voltage: {status.voltage_kv}")
            return False
        
        # Check capacity values are positive
        if status.total_capacity_mw <= 0 or status.available_capacity_mw < 0:
            logger.warning(f"Invalid capacity values: total={status.total_capacity_mw}, available={status.available_capacity_mw}")
            return False
        
        # Check load factor is reasonable
        if not (0 <= status.load_factor <= 1):
            logger.warning(f"Invalid load factor: {status.load_factor}")
            return False
        
        return True
    
    def process_all_operators(self):
        """Process data from all configured grid operators"""
        operators = self.config['data_sources']['grid_operators']
        
        for operator_config in operators:
            logger.info(f"Processing data from {operator_config['name']}")
            
            status = self.fetch_grid_status(operator_config)
            if status and self.validate_grid_status(status):
                # Store in S3
                self._store_grid_status_s3(status)
                
                # Send to Kafka
                self._send_to_kafka(status)
                
                logger.info(f"Successfully processed {operator_config['name']} data")
            else:
                logger.warning(f"Failed to process {operator_config['name']} data")
    
    def _store_grid_status_s3(self, status: GridStatus):
        """Store grid status to S3"""
        s3_key = f"raw/grid_status/{status.timestamp.strftime('%Y/%m/%d')}/{status.operator_name}_{status.timestamp.isoformat()}.json"
        
        status_dict = {
            "operator_name": status.operator_name,
            "timestamp": status.timestamp.isoformat(),
            "total_capacity_mw": status.total_capacity_mw,
            "available_capacity_mw": status.available_capacity_mw,
            "load_factor": status.load_factor,
            "frequency_hz": status.frequency_hz,
            "voltage_kv": status.voltage_kv,
            "grid_stability_score": status.grid_stability_score,
            "renewable_percentage": status.renewable_percentage,
            "region": status.region,
            "processing_timestamp": datetime.utcnow().isoformat()
        }
        
        self.s3_client.put_object(
            Bucket=self.config['aws']['s3_bucket'],
            Key=s3_key,
            Body=json.dumps(status_dict),
            ContentType='application/json'
        )
    
    def _send_to_kafka(self, status: GridStatus):
        """Send grid status to Kafka topic"""
        kafka_data = {
            "operator_name": status.operator_name,
            "timestamp": status.timestamp.isoformat(),
            "total_capacity_mw": status.total_capacity_mw,
            "available_capacity_mw": status.available_capacity_mw,
            "load_factor": status.load_factor,
            "frequency_hz": status.frequency_hz,
            "voltage_kv": status.voltage_kv,
            "grid_stability_score": status.grid_stability_score,
            "renewable_percentage": status.renewable_percentage,
            "region": status.region
        }
        
        self.kafka_producer.send(
            self.config['kafka']['topics']['grid_status'],
            value=kafka_data
        )
    
    def run_continuous_ingestion(self, interval_seconds: int = 300):
        """Run continuous ingestion with specified interval"""
        logger.info(f"Starting continuous grid operator ingestion (interval: {interval_seconds}s)")
        
        try:
            while True:
                self.process_all_operators()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("Stopping continuous grid operator ingestion")

def main():
    """Main function to run the grid operator ingestion"""
    pipeline = GridOperatorIngestionPipeline()
    
    # Run continuous ingestion
    pipeline.run_continuous_ingestion()

if __name__ == "__main__":
    main()
