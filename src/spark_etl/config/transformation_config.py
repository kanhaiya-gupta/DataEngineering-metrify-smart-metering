"""
Transformation Configuration Service
Loads and manages transformation rules from YAML configuration files
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from ...core.config.config_loader import config_loader

logger = logging.getLogger(__name__)


@dataclass
class BusinessRule:
    """Represents a business rule with its parameters"""
    name: str
    parameters: Dict[str, Any]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a parameter value with default"""
        return self.parameters.get(key, default)


@dataclass
class TransformationConfig:
    """Configuration for data transformations"""
    business_rules: Dict[str, Dict[str, BusinessRule]]
    anomaly_detection: Dict[str, Any]
    time_series: Dict[str, Any]
    predictive: Dict[str, Any]
    environment_overrides: Dict[str, Dict[str, Any]]
    
    def get_business_rule(self, category: str, rule_name: str) -> Optional[BusinessRule]:
        """Get a specific business rule"""
        if category in self.business_rules and rule_name in self.business_rules[category]:
            return self.business_rules[category][rule_name]
        return None
    
    def get_anomaly_threshold(self, key: str, default: Any = None) -> Any:
        """Get anomaly detection threshold"""
        return self.anomaly_detection.get(key, default)
    
    def get_time_series_config(self, key: str, default: Any = None) -> Any:
        """Get time-series configuration"""
        return self.time_series.get(key, default)
    
    def get_predictive_config(self, key: str, default: Any = None) -> Any:
        """Get predictive configuration"""
        return self.predictive.get(key, default)
    
    def apply_environment_overrides(self, environment: str) -> 'TransformationConfig':
        """Apply environment-specific overrides"""
        if environment not in self.environment_overrides:
            return self
        
        overrides = self.environment_overrides[environment]
        
        # Deep merge overrides
        import copy
        config_copy = copy.deepcopy(self)
        
        # Apply overrides recursively
        self._apply_overrides(config_copy, overrides)
        
        return config_copy
    
    def _apply_overrides(self, config: 'TransformationConfig', overrides: Dict[str, Any]):
        """Recursively apply overrides to configuration"""
        for key, value in overrides.items():
            if hasattr(config, key):
                if isinstance(value, dict) and isinstance(getattr(config, key), dict):
                    # Merge dictionaries
                    getattr(config, key).update(value)
                else:
                    # Replace value
                    setattr(config, key, value)


class TransformationConfigService:
    """Service for loading and managing transformation configurations"""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self._config_cache = {}
    
    def get_smart_meter_config(self) -> TransformationConfig:
        """Get smart meter transformation configuration"""
        cache_key = f"smart_meters_{self.environment}"
        if cache_key not in self._config_cache:
            self._config_cache[cache_key] = self._load_config("smart_meters")
        return self._config_cache[cache_key]
    
    def get_grid_operator_config(self) -> TransformationConfig:
        """Get grid operator transformation configuration"""
        cache_key = f"grid_operators_{self.environment}"
        if cache_key not in self._config_cache:
            self._config_cache[cache_key] = self._load_config("grid_operators")
        return self._config_cache[cache_key]
    
    def get_weather_station_config(self) -> TransformationConfig:
        """Get weather station transformation configuration"""
        cache_key = f"weather_stations_{self.environment}"
        if cache_key not in self._config_cache:
            self._config_cache[cache_key] = self._load_config("weather_stations")
        return self._config_cache[cache_key]
    
    def _load_config(self, data_source: str) -> TransformationConfig:
        """Load transformation configuration for a data source"""
        try:
            # Load data sources configuration
            data_sources_config = config_loader.get_data_sources_config()
            
            # Get the specific data source config
            if data_source == "smart_meters":
                source_config = data_sources_config.smart_meters
            elif data_source == "grid_operators":
                source_config = data_sources_config.grid_operators
            elif data_source == "weather_stations":
                source_config = data_sources_config.weather_stations
            else:
                raise ValueError(f"Unknown data source: {data_source}")
            
            # Extract transformation configuration
            transformations = getattr(source_config, 'transformations', {})
            
            # Parse business rules
            business_rules = {}
            if 'business_rules' in transformations:
                for category, rules in transformations['business_rules'].items():
                    business_rules[category] = {}
                    for rule_name, rule_params in rules.items():
                        business_rules[category][rule_name] = BusinessRule(
                            name=rule_name,
                            parameters=rule_params
                        )
            
            # Create transformation config
            config = TransformationConfig(
                business_rules=business_rules,
                anomaly_detection=transformations.get('anomaly_detection', {}),
                time_series=transformations.get('time_series', {}),
                predictive=transformations.get('predictive', {}),
                environment_overrides=transformations.get('environment_overrides', {})
            )
            
            # Apply environment overrides
            config = config.apply_environment_overrides(self.environment)
            
            logger.info(f"Loaded transformation configuration for {data_source} in {self.environment} environment")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load transformation configuration for {data_source}: {str(e)}")
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> TransformationConfig:
        """Get default transformation configuration"""
        return TransformationConfig(
            business_rules={},
            anomaly_detection={'zscore_threshold': 3.0, 'window_size': 20},
            time_series={'windows': {'short_term': 24, 'medium_term': 168}},
            predictive={},
            environment_overrides={}
        )


# Global instance
_transformation_config_service = None


def get_transformation_config_service(environment: str = "development") -> TransformationConfigService:
    """Get the global transformation configuration service instance"""
    global _transformation_config_service
    if _transformation_config_service is None or _transformation_config_service.environment != environment:
        _transformation_config_service = TransformationConfigService(environment)
    return _transformation_config_service
