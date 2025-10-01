"""
Data Quality Validation Framework
Comprehensive data quality checks for all data sources
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import yaml
from pathlib import Path
import boto3
from snowflake.connector import connect
import great_expectations as ge
from great_expectations.core import ExpectationSuite
from great_expectations.dataset import PandasDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QualityCheckResult:
    """Data class for quality check results"""
    check_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    timestamp: datetime

class DataQualityValidator:
    """Main class for data quality validation"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data quality validator"""
        self.config = self._load_config(config_path)
        self.snowflake_conn = self._init_snowflake_connection()
        self.s3_client = boto3.client('s3')
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _init_snowflake_connection(self):
        """Initialize Snowflake connection"""
        db_config = self.config['databases']['snowflake']
        return connect(
            user=db_config['user'],
            password=db_config['password'],
            account=db_config['account'],
            warehouse=db_config['warehouse'],
            database=db_config['database'],
            schema=db_config['schema']
        )
    
    def validate_smart_meter_data(self, date: str) -> bool:
        """Validate smart meter data quality"""
        logger.info(f"Validating smart meter data for date: {date}")
        
        try:
            # Query smart meter data
            query = f"""
            SELECT 
                meter_id,
                timestamp,
                consumption_kwh,
                voltage,
                current,
                power_factor,
                frequency,
                data_quality_score
            FROM raw.smart_meter_readings 
            WHERE DATE(timestamp) = '{date}'
            """
            
            df = pd.read_sql(query, self.snowflake_conn)
            
            if df.empty:
                logger.warning(f"No smart meter data found for date: {date}")
                return False
            
            # Run quality checks
            checks = [
                self._check_completeness(df),
                self._check_validity(df),
                self._check_consistency(df),
                self._check_accuracy(df),
                self._check_timeliness(df)
            ]
            
            # Calculate overall quality score
            passed_checks = sum(1 for check in checks if check.passed)
            overall_score = passed_checks / len(checks)
            
            logger.info(f"Smart meter data quality score: {overall_score:.2f}")
            
            # Store quality results
            self._store_quality_results('smart_meter_readings', date, checks, overall_score)
            
            return overall_score >= self.config['data_quality']['smart_meter_readings']['threshold']
            
        except Exception as e:
            logger.error(f"Error validating smart meter data: {e}")
            return False
    
    def validate_grid_operator_data(self, date: str) -> bool:
        """Validate grid operator data quality"""
        logger.info(f"Validating grid operator data for date: {date}")
        
        try:
            # Query grid operator data
            query = f"""
            SELECT 
                operator_name,
                timestamp,
                total_capacity_mw,
                available_capacity_mw,
                load_factor,
                frequency_hz,
                voltage_kv,
                grid_stability_score
            FROM raw.grid_status 
            WHERE DATE(timestamp) = '{date}'
            """
            
            df = pd.read_sql(query, self.snowflake_conn)
            
            if df.empty:
                logger.warning(f"No grid operator data found for date: {date}")
                return False
            
            # Run quality checks
            checks = [
                self._check_completeness(df),
                self._check_validity(df),
                self._check_consistency(df),
                self._check_accuracy(df),
                self._check_timeliness(df)
            ]
            
            # Calculate overall quality score
            passed_checks = sum(1 for check in checks if check.passed)
            overall_score = passed_checks / len(checks)
            
            logger.info(f"Grid operator data quality score: {overall_score:.2f}")
            
            # Store quality results
            self._store_quality_results('grid_status', date, checks, overall_score)
            
            return overall_score >= 0.8  # Default threshold for grid data
            
        except Exception as e:
            logger.error(f"Error validating grid operator data: {e}")
            return False
    
    def validate_weather_data(self, date: str) -> bool:
        """Validate weather data quality"""
        logger.info(f"Validating weather data for date: {date}")
        
        try:
            # Query weather data
            query = f"""
            SELECT 
                city,
                timestamp,
                temperature_celsius,
                humidity_percent,
                pressure_hpa,
                wind_speed_ms,
                wind_direction_degrees,
                cloud_cover_percent
            FROM raw.weather_data 
            WHERE DATE(timestamp) = '{date}'
            """
            
            df = pd.read_sql(query, self.snowflake_conn)
            
            if df.empty:
                logger.warning(f"No weather data found for date: {date}")
                return False
            
            # Run quality checks
            checks = [
                self._check_completeness(df),
                self._check_validity(df),
                self._check_consistency(df),
                self._check_accuracy(df),
                self._check_timeliness(df)
            ]
            
            # Calculate overall quality score
            passed_checks = sum(1 for check in checks if check.passed)
            overall_score = passed_checks / len(checks)
            
            logger.info(f"Weather data quality score: {overall_score:.2f}")
            
            # Store quality results
            self._store_quality_results('weather_data', date, checks, overall_score)
            
            return overall_score >= 0.8  # Default threshold for weather data
            
        except Exception as e:
            logger.error(f"Error validating weather data: {e}")
            return False
    
    def _check_completeness(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check data completeness"""
        total_rows = len(df)
        null_counts = df.isnull().sum()
        completeness_scores = []
        
        for column in df.columns:
            if column in ['meter_id', 'timestamp', 'operator_name', 'city']:
                # Critical fields should not be null
                null_count = null_counts[column]
                completeness_score = (total_rows - null_count) / total_rows
                completeness_scores.append(completeness_score)
            else:
                # Other fields can have some nulls
                null_count = null_counts[column]
                completeness_score = max(0, (total_rows - null_count) / total_rows)
                completeness_scores.append(completeness_score)
        
        overall_completeness = np.mean(completeness_scores)
        passed = overall_completeness >= 0.95
        
        return QualityCheckResult(
            check_name="completeness",
            passed=passed,
            score=overall_completeness,
            details={
                "total_rows": total_rows,
                "null_counts": null_counts.to_dict(),
                "column_scores": dict(zip(df.columns, completeness_scores))
            },
            timestamp=datetime.utcnow()
        )
    
    def _check_validity(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check data validity against business rules"""
        validity_checks = []
        
        # Smart meter specific checks
        if 'consumption_kwh' in df.columns:
            valid_consumption = (df['consumption_kwh'] >= 0) & (df['consumption_kwh'] <= 1000)
            validity_checks.append(valid_consumption.mean())
        
        if 'voltage' in df.columns:
            valid_voltage = (df['voltage'] >= 200) & (df['voltage'] <= 250)
            validity_checks.append(valid_voltage.mean())
        
        if 'current' in df.columns:
            valid_current = (df['current'] >= 0) & (df['current'] <= 100)
            validity_checks.append(valid_current.mean())
        
        if 'power_factor' in df.columns:
            valid_power_factor = (df['power_factor'] >= 0) & (df['power_factor'] <= 1)
            validity_checks.append(valid_power_factor.mean())
        
        if 'frequency' in df.columns:
            valid_frequency = (df['frequency'] >= 49.5) & (df['frequency'] <= 50.5)
            validity_checks.append(valid_frequency.mean())
        
        # Grid operator specific checks
        if 'total_capacity_mw' in df.columns:
            valid_capacity = df['total_capacity_mw'] > 0
            validity_checks.append(valid_capacity.mean())
        
        if 'load_factor' in df.columns:
            valid_load_factor = (df['load_factor'] >= 0) & (df['load_factor'] <= 1)
            validity_checks.append(valid_load_factor.mean())
        
        if 'frequency_hz' in df.columns:
            valid_frequency_hz = (df['frequency_hz'] >= 49.5) & (df['frequency_hz'] <= 50.5)
            validity_checks.append(valid_frequency_hz.mean())
        
        if 'voltage_kv' in df.columns:
            valid_voltage_kv = (df['voltage_kv'] >= 380) & (df['voltage_kv'] <= 420)
            validity_checks.append(valid_voltage_kv.mean())
        
        # Weather specific checks
        if 'temperature_celsius' in df.columns:
            valid_temperature = (df['temperature_celsius'] >= -30) & (df['temperature_celsius'] <= 50)
            validity_checks.append(valid_temperature.mean())
        
        if 'humidity_percent' in df.columns:
            valid_humidity = (df['humidity_percent'] >= 0) & (df['humidity_percent'] <= 100)
            validity_checks.append(valid_humidity.mean())
        
        if 'pressure_hpa' in df.columns:
            valid_pressure = (df['pressure_hpa'] >= 950) & (df['pressure_hpa'] <= 1050)
            validity_checks.append(valid_pressure.mean())
        
        if 'wind_speed_ms' in df.columns:
            valid_wind_speed = (df['wind_speed_ms'] >= 0) & (df['wind_speed_ms'] <= 50)
            validity_checks.append(valid_wind_speed.mean())
        
        if 'wind_direction_degrees' in df.columns:
            valid_wind_direction = (df['wind_direction_degrees'] >= 0) & (df['wind_direction_degrees'] <= 360)
            validity_checks.append(valid_wind_direction.mean())
        
        overall_validity = np.mean(validity_checks) if validity_checks else 1.0
        passed = overall_validity >= 0.95
        
        return QualityCheckResult(
            check_name="validity",
            passed=passed,
            score=overall_validity,
            details={
                "validity_checks": len(validity_checks),
                "overall_validity": overall_validity
            },
            timestamp=datetime.utcnow()
        )
    
    def _check_consistency(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check data consistency"""
        consistency_checks = []
        
        # Check for duplicate records
        if 'meter_id' in df.columns and 'timestamp' in df.columns:
            duplicates = df.duplicated(subset=['meter_id', 'timestamp']).sum()
            consistency_checks.append(1 - (duplicates / len(df)))
        
        if 'operator_name' in df.columns and 'timestamp' in df.columns:
            duplicates = df.duplicated(subset=['operator_name', 'timestamp']).sum()
            consistency_checks.append(1 - (duplicates / len(df)))
        
        if 'city' in df.columns and 'timestamp' in df.columns:
            duplicates = df.duplicated(subset=['city', 'timestamp']).sum()
            consistency_checks.append(1 - (duplicates / len(df)))
        
        # Check for logical consistency
        if 'consumption_kwh' in df.columns and 'voltage' in df.columns and 'current' in df.columns:
            # Check if consumption is reasonable given voltage and current
            theoretical_max = df['voltage'] * df['current'] / 1000  # Convert to kWh
            reasonable_consumption = (df['consumption_kwh'] <= theoretical_max * 1.1).mean()
            consistency_checks.append(reasonable_consumption)
        
        if 'total_capacity_mw' in df.columns and 'available_capacity_mw' in df.columns:
            # Check if available capacity is not greater than total capacity
            logical_capacity = (df['available_capacity_mw'] <= df['total_capacity_mw']).mean()
            consistency_checks.append(logical_capacity)
        
        overall_consistency = np.mean(consistency_checks) if consistency_checks else 1.0
        passed = overall_consistency >= 0.95
        
        return QualityCheckResult(
            check_name="consistency",
            passed=passed,
            score=overall_consistency,
            details={
                "consistency_checks": len(consistency_checks),
                "overall_consistency": overall_consistency
            },
            timestamp=datetime.utcnow()
        )
    
    def _check_accuracy(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check data accuracy using statistical methods"""
        accuracy_checks = []
        
        # Check for outliers using IQR method
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in ['meter_id', 'timestamp']:
                continue
                
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            accuracy_score = 1 - (outliers / len(df))
            accuracy_checks.append(accuracy_score)
        
        overall_accuracy = np.mean(accuracy_checks) if accuracy_checks else 1.0
        passed = overall_accuracy >= 0.9
        
        return QualityCheckResult(
            check_name="accuracy",
            passed=passed,
            score=overall_accuracy,
            details={
                "accuracy_checks": len(accuracy_checks),
                "overall_accuracy": overall_accuracy
            },
            timestamp=datetime.utcnow()
        )
    
    def _check_timeliness(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check data timeliness"""
        if 'timestamp' not in df.columns:
            return QualityCheckResult(
                check_name="timeliness",
                passed=True,
                score=1.0,
                details={"message": "No timestamp column found"},
                timestamp=datetime.utcnow()
            )
        
        # Check if data is recent (within last 24 hours)
        current_time = datetime.utcnow()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        time_diffs = (current_time - df['timestamp']).dt.total_seconds() / 3600  # Convert to hours
        
        recent_data = (time_diffs <= 24).mean()
        passed = recent_data >= 0.8
        
        return QualityCheckResult(
            check_name="timeliness",
            passed=passed,
            score=recent_data,
            details={
                "recent_data_percentage": recent_data,
                "max_age_hours": time_diffs.max(),
                "avg_age_hours": time_diffs.mean()
            },
            timestamp=datetime.utcnow()
        )
    
    def _store_quality_results(self, table_name: str, date: str, checks: List[QualityCheckResult], overall_score: float):
        """Store quality check results in S3"""
        try:
            results = {
                "table_name": table_name,
                "date": date,
                "overall_score": overall_score,
                "checks": [
                    {
                        "check_name": check.check_name,
                        "passed": check.passed,
                        "score": check.score,
                        "details": check.details,
                        "timestamp": check.timestamp.isoformat()
                    }
                    for check in checks
                ],
                "processed_at": datetime.utcnow().isoformat()
            }
            
            s3_key = f"quality_results/{table_name}/{date}/quality_report.json"
            
            self.s3_client.put_object(
                Bucket=self.config['aws']['s3_bucket'],
                Key=s3_key,
                Body=json.dumps(results, indent=2),
                ContentType='application/json'
            )
            
            logger.info(f"Quality results stored for {table_name} on {date}")
            
        except Exception as e:
            logger.error(f"Error storing quality results: {e}")

def main():
    """Main function to run data quality validation"""
    validator = DataQualityValidator()
    
    # Get date from command line or use today
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y-%m-%d')
    
    # Run validation for all data sources
    smart_meter_quality = validator.validate_smart_meter_data(date)
    grid_quality = validator.validate_grid_operator_data(date)
    weather_quality = validator.validate_weather_data(date)
    
    # Overall result
    overall_quality = all([smart_meter_quality, grid_quality, weather_quality])
    
    if overall_quality:
        logger.info("All data quality checks passed")
        sys.exit(0)
    else:
        logger.error("Data quality checks failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
