"""
MLflow Setup and Configuration

This module provides MLflow configuration and utilities for the smart meter ML pipeline.
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import mlflow
import mlflow.tensorflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

logger = logging.getLogger(__name__)


class MLflowManager:
    """
    MLflow management class for experiment tracking, model registry, and deployment
    """
    
    def __init__(self, 
                 tracking_uri: str = "sqlite:///mlflow.db",
                 registry_uri: str = "sqlite:///mlflow_registry.db",
                 experiment_name: str = "smart_meter_ml"):
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.experiment_name = experiment_name
        self.client = None
        
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking and registry"""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Set registry URI
            mlflow.set_registry_uri(self.registry_uri)
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=self.experiment_name,
                    artifact_location=f"mlruns/{self.experiment_name}"
                )
                logger.info(f"Created experiment: {self.experiment_name} with ID: {experiment_id}")
            else:
                logger.info(f"Using existing experiment: {self.experiment_name}")
            
            # Set active experiment
            mlflow.set_experiment(self.experiment_name)
            
            # Initialize client
            self.client = MlflowClient(tracking_uri=self.tracking_uri)
            
            logger.info("MLflow setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {str(e)}")
            raise
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Start a new MLflow run"""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return mlflow.start_run(run_name=run_name, tags=tags)
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log parameters to current run"""
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to current run"""
        mlflow.log_metrics(metrics)
        logger.info(f"Logged {len(metrics)} metrics")
    
    def log_artifacts(self, artifact_path: str, artifact_dir: Optional[str] = None):
        """Log artifacts to current run"""
        if artifact_dir:
            mlflow.log_artifacts(artifact_dir)
        else:
            mlflow.log_artifact(artifact_path)
        logger.info(f"Logged artifact: {artifact_path}")
    
    def log_model(self, model, artifact_path: str, 
                  model_type: str = "tensorflow",
                  registered_model_name: Optional[str] = None):
        """Log model to current run and optionally register it"""
        if model_type == "tensorflow":
            mlflow.tensorflow.log_model(
                model, 
                artifact_path,
                registered_model_name=registered_model_name
            )
        elif model_type == "sklearn":
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"Logged model: {artifact_path}")
    
    def register_model(self, run_id: str, model_name: str, 
                      model_path: str = "model",
                      version: Optional[str] = None):
        """Register model in MLflow Model Registry"""
        try:
            model_uri = f"runs:/{run_id}/{model_path}"
            
            if version:
                registered_model = mlflow.register_model(
                    model_uri, 
                    model_name,
                    tags={"version": version}
                )
            else:
                registered_model = mlflow.register_model(model_uri, model_name)
            
            logger.info(f"Registered model: {model_name} with version: {registered_model.version}")
            return registered_model
            
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            raise
    
    def get_model_versions(self, model_name: str) -> list:
        """Get all versions of a registered model"""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            return [v for v in versions]
        except Exception as e:
            logger.error(f"Failed to get model versions: {str(e)}")
            return []
    
    def get_latest_model_version(self, model_name: str) -> Optional[Any]:
        """Get the latest version of a registered model"""
        versions = self.get_model_versions(model_name)
        if not versions:
            return None
        
        # Sort by version number and return latest
        latest_version = max(versions, key=lambda x: int(x.version))
        return latest_version
    
    def promote_model_to_staging(self, model_name: str, version: str):
        """Promote model to staging"""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging"
            )
            logger.info(f"Promoted model {model_name} version {version} to Staging")
        except Exception as e:
            logger.error(f"Failed to promote model to staging: {str(e)}")
            raise
    
    def promote_model_to_production(self, model_name: str, version: str):
        """Promote model to production"""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            logger.info(f"Promoted model {model_name} version {version} to Production")
        except Exception as e:
            logger.error(f"Failed to promote model to production: {str(e)}")
            raise
    
    def get_production_model(self, model_name: str) -> Optional[Any]:
        """Get the production model"""
        try:
            production_models = self.client.get_latest_versions(
                model_name, 
                stages=["Production"]
            )
            return production_models[0] if production_models else None
        except Exception as e:
            logger.error(f"Failed to get production model: {str(e)}")
            return None
    
    def compare_models(self, model_name: str, versions: list) -> Dict[str, Any]:
        """Compare different versions of a model"""
        try:
            comparison = {}
            
            for version in versions:
                version_info = self.client.get_model_version(model_name, version)
                run = self.client.get_run(version_info.run_id)
                
                comparison[version] = {
                    "version": version_info.version,
                    "stage": version_info.current_stage,
                    "creation_timestamp": version_info.creation_timestamp,
                    "metrics": run.data.metrics,
                    "parameters": run.data.params
                }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models: {str(e)}")
            return {}
    
    def search_runs(self, experiment_id: Optional[str] = None, 
                   filter_string: Optional[str] = None,
                   max_results: int = 100) -> list:
        """Search runs in MLflow"""
        try:
            runs = self.client.search_runs(
                experiment_ids=[experiment_id] if experiment_id else None,
                filter_string=filter_string,
                max_results=max_results,
                order_by=["start_time DESC"]
            )
            return runs
        except Exception as e:
            logger.error(f"Failed to search runs: {str(e)}")
            return []
    
    def get_best_run(self, experiment_id: Optional[str] = None,
                    metric_name: str = "f1_score",
                    ascending: bool = False) -> Optional[Any]:
        """Get the best run based on a metric"""
        try:
            runs = self.search_runs(experiment_id)
            if not runs:
                return None
            
            # Filter runs that have the metric
            runs_with_metric = [
                run for run in runs 
                if metric_name in run.data.metrics
            ]
            
            if not runs_with_metric:
                return None
            
            # Sort by metric value
            best_run = sorted(
                runs_with_metric,
                key=lambda x: x.data.metrics[metric_name],
                reverse=not ascending
            )[0]
            
            return best_run
            
        except Exception as e:
            logger.error(f"Failed to get best run: {str(e)}")
            return None
    
    def create_model_comparison_report(self, model_name: str, 
                                     output_path: str = "model_comparison.html"):
        """Create an HTML report comparing model versions"""
        try:
            versions = self.get_model_versions(model_name)
            if not versions:
                logger.warning(f"No versions found for model: {model_name}")
                return
            
            comparison = self.compare_models(model_name, [v.version for v in versions])
            
            # Create HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Model Comparison Report - {model_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .metric {{ font-weight: bold; color: #2c5aa0; }}
                </style>
            </head>
            <body>
                <h1>Model Comparison Report</h1>
                <h2>Model: {model_name}</h2>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h3>Version Comparison</h3>
                <table>
                    <tr>
                        <th>Version</th>
                        <th>Stage</th>
                        <th>Created</th>
                        <th>F1 Score</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                    </tr>
            """
            
            for version, info in comparison.items():
                metrics = info.get('metrics', {})
                html_content += f"""
                    <tr>
                        <td>{version}</td>
                        <td>{info.get('stage', 'N/A')}</td>
                        <td>{datetime.fromtimestamp(info.get('creation_timestamp', 0) / 1000).strftime('%Y-%m-%d %H:%M')}</td>
                        <td class="metric">{metrics.get('f1_score', 'N/A'):.4f if isinstance(metrics.get('f1_score'), (int, float)) else 'N/A'}</td>
                        <td class="metric">{metrics.get('accuracy', 'N/A'):.4f if isinstance(metrics.get('accuracy'), (int, float)) else 'N/A'}</td>
                        <td class="metric">{metrics.get('precision', 'N/A'):.4f if isinstance(metrics.get('precision'), (int, float)) else 'N/A'}</td>
                        <td class="metric">{metrics.get('recall', 'N/A'):.4f if isinstance(metrics.get('recall'), (int, float)) else 'N/A'}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </body>
            </html>
            """
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Model comparison report saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create model comparison report: {str(e)}")
            raise
    
    def cleanup_old_runs(self, experiment_id: Optional[str] = None,
                        max_age_days: int = 30):
        """Clean up old runs to save storage space"""
        try:
            runs = self.search_runs(experiment_id)
            cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 3600)
            
            deleted_count = 0
            for run in runs:
                if run.info.start_time < cutoff_date * 1000:  # Convert to milliseconds
                    self.client.delete_run(run.info.run_id)
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old runs")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old runs: {str(e)}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of the current experiment"""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            runs = self.search_runs(experiment.experiment_id)
            
            summary = {
                "experiment_name": self.experiment_name,
                "experiment_id": experiment.experiment_id,
                "total_runs": len(runs),
                "active_runs": len([r for r in runs if r.info.status == "RUNNING"]),
                "finished_runs": len([r for r in runs if r.info.status == "FINISHED"]),
                "failed_runs": len([r for r in runs if r.info.status == "FAILED"]),
                "last_run": max(runs, key=lambda x: x.info.start_time).info.start_time if runs else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get experiment summary: {str(e)}")
            return {}


# Global MLflow manager instance
mlflow_manager = MLflowManager()


def get_mlflow_manager() -> MLflowManager:
    """Get the global MLflow manager instance"""
    return mlflow_manager
