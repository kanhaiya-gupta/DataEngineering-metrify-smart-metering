"""
Model Registry

This module provides model registry capabilities:
- Model versioning and metadata management
- Model lifecycle management
- Model deployment tracking
- Model performance monitoring
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import shutil

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model status in registry"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ModelStage(Enum):
    """Model deployment stage"""
    NONE = "none"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelMetadata:
    """Model metadata"""
    name: str
    version: str
    description: str
    model_type: str
    framework: str
    created_at: datetime
    created_by: str
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    size_bytes: int = 0
    checksum: str = ""


@dataclass
class ModelVersion:
    """Model version information"""
    name: str
    version: str
    status: ModelStatus
    stage: ModelStage
    metadata: ModelMetadata
    model_path: str
    created_at: datetime
    updated_at: datetime
    deployment_info: Dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """
    Model Registry for managing ML models and their versions
    """

    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = registry_path
        self.models: Dict[str, Dict[str, ModelVersion]] = {}  # name -> version -> ModelVersion
        self.current_stages: Dict[str, ModelStage] = {}  # name -> current stage
        
        # Create registry directory if it doesn't exist
        os.makedirs(registry_path, exist_ok=True)
        
        # Load existing models
        self._load_registry()

    def register_model(self,
                      name: str,
                      version: str,
                      model_path: str,
                      description: str = "",
                      model_type: str = "unknown",
                      framework: str = "unknown",
                      created_by: str = "system",
                      tags: Optional[List[str]] = None,
                      parameters: Optional[Dict[str, Any]] = None,
                      metrics: Optional[Dict[str, float]] = None,
                      dependencies: Optional[List[str]] = None) -> ModelVersion:
        """
        Register a new model version
        
        Args:
            name: Model name
            version: Model version
            model_path: Path to model file
            description: Model description
            model_type: Type of model (e.g., 'classification', 'regression')
            framework: ML framework used (e.g., 'tensorflow', 'pytorch')
            created_by: User who created the model
            tags: List of tags
            parameters: Model parameters
            metrics: Model performance metrics
            dependencies: List of dependencies
            
        Returns:
            Registered model version
        """
        # Validate model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Calculate file size and checksum
        file_size = os.path.getsize(model_path)
        checksum = self._calculate_checksum(model_path)
        
        # Create metadata
        metadata = ModelMetadata(
            name=name,
            version=version,
            description=description,
            model_type=model_type,
            framework=framework,
            created_at=datetime.utcnow(),
            created_by=created_by,
            tags=tags or [],
            parameters=parameters or {},
            metrics=metrics or {},
            dependencies=dependencies or [],
            size_bytes=file_size,
            checksum=checksum
        )
        
        # Create model version
        model_version = ModelVersion(
            name=name,
            version=version,
            status=ModelStatus.DEVELOPMENT,
            stage=ModelStage.NONE,
            metadata=metadata,
            model_path=model_path,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Store in registry
        if name not in self.models:
            self.models[name] = {}
        
        self.models[name][version] = model_version
        
        # Save to disk
        self._save_model_version(model_version)
        
        logger.info(f"Registered model {name} version {version}")
        return model_version

    def get_model(self, name: str, version: Optional[str] = None) -> Optional[ModelVersion]:
        """
        Get model version
        
        Args:
            name: Model name
            version: Model version (if None, returns latest)
            
        Returns:
            Model version or None if not found
        """
        if name not in self.models:
            return None
        
        if version is None:
            # Return latest version
            versions = list(self.models[name].keys())
            if not versions:
                return None
            version = max(versions)  # Simple version comparison
        
        return self.models[name].get(version)

    def list_models(self) -> List[str]:
        """List all model names"""
        return list(self.models.keys())

    def list_versions(self, name: str) -> List[str]:
        """List all versions for a model"""
        if name not in self.models:
            return []
        return list(self.models[name].keys())

    def update_model_status(self, name: str, version: str, status: ModelStatus) -> bool:
        """
        Update model status
        
        Args:
            name: Model name
            version: Model version
            status: New status
            
        Returns:
            True if updated successfully
        """
        model_version = self.get_model(name, version)
        if not model_version:
            logger.error(f"Model {name} version {version} not found")
            return False
        
        model_version.status = status
        model_version.updated_at = datetime.utcnow()
        
        self._save_model_version(model_version)
        logger.info(f"Updated model {name} version {version} status to {status.value}")
        return True

    def promote_model(self, name: str, version: str, stage: ModelStage) -> bool:
        """
        Promote model to a deployment stage
        
        Args:
            name: Model name
            version: Model version
            stage: Target stage
            
        Returns:
            True if promoted successfully
        """
        model_version = self.get_model(name, version)
        if not model_version:
            logger.error(f"Model {name} version {version} not found")
            return False
        
        # Update model stage
        model_version.stage = stage
        model_version.updated_at = datetime.utcnow()
        
        # Update current stage for the model
        self.current_stages[name] = stage
        
        # Add deployment info
        model_version.deployment_info = {
            "promoted_at": datetime.utcnow().isoformat(),
            "promoted_to": stage.value
        }
        
        self._save_model_version(model_version)
        logger.info(f"Promoted model {name} version {version} to {stage.value}")
        return True

    def get_production_model(self, name: str) -> Optional[ModelVersion]:
        """
        Get the current production model
        
        Args:
            name: Model name
            
        Returns:
            Production model version or None
        """
        if name not in self.models:
            return None
        
        # Find production model
        for version, model_version in self.models[name].items():
            if model_version.stage == ModelStage.PRODUCTION:
                return model_version
        
        return None

    def archive_model(self, name: str, version: str) -> bool:
        """
        Archive a model version
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            True if archived successfully
        """
        model_version = self.get_model(name, version)
        if not model_version:
            logger.error(f"Model {name} version {version} not found")
            return False
        
        # Update status and stage
        model_version.status = ModelStatus.ARCHIVED
        model_version.stage = ModelStage.ARCHIVED
        model_version.updated_at = datetime.utcnow()
        
        self._save_model_version(model_version)
        logger.info(f"Archived model {name} version {version}")
        return True

    def delete_model(self, name: str, version: str) -> bool:
        """
        Delete a model version
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            True if deleted successfully
        """
        if name not in self.models or version not in self.models[name]:
            logger.error(f"Model {name} version {version} not found")
            return False
        
        model_version = self.models[name][version]
        
        # Delete model file
        if os.path.exists(model_version.model_path):
            os.remove(model_version.model_path)
        
        # Remove from registry
        del self.models[name][version]
        
        # Remove model if no versions left
        if not self.models[name]:
            del self.models[name]
            if name in self.current_stages:
                del self.current_stages[name]
        
        # Delete registry file
        registry_file = os.path.join(self.registry_path, f"{name}_{version}.json")
        if os.path.exists(registry_file):
            os.remove(registry_file)
        
        logger.info(f"Deleted model {name} version {version}")
        return True

    def search_models(self, 
                     name_pattern: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     model_type: Optional[str] = None,
                     status: Optional[ModelStatus] = None,
                     stage: Optional[ModelStage] = None) -> List[ModelVersion]:
        """
        Search for models based on criteria
        
        Args:
            name_pattern: Pattern to match model names
            tags: List of tags to match
            model_type: Model type to match
            status: Model status to match
            stage: Model stage to match
            
        Returns:
            List of matching model versions
        """
        results = []
        
        for model_name, versions in self.models.items():
            for version, model_version in versions.items():
                # Check name pattern
                if name_pattern and name_pattern.lower() not in model_name.lower():
                    continue
                
                # Check tags
                if tags and not any(tag in model_version.metadata.tags for tag in tags):
                    continue
                
                # Check model type
                if model_type and model_version.metadata.model_type != model_type:
                    continue
                
                # Check status
                if status and model_version.status != status:
                    continue
                
                # Check stage
                if stage and model_version.stage != stage:
                    continue
                
                results.append(model_version)
        
        return results

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _save_model_version(self, model_version: ModelVersion):
        """Save model version to disk"""
        registry_file = os.path.join(
            self.registry_path, 
            f"{model_version.name}_{model_version.version}.json"
        )
        
        # Convert to dict for JSON serialization
        data = {
            "name": model_version.name,
            "version": model_version.version,
            "status": model_version.status.value,
            "stage": model_version.stage.value,
            "metadata": {
                "name": model_version.metadata.name,
                "version": model_version.metadata.version,
                "description": model_version.metadata.description,
                "model_type": model_version.metadata.model_type,
                "framework": model_version.metadata.framework,
                "created_at": model_version.metadata.created_at.isoformat(),
                "created_by": model_version.metadata.created_by,
                "tags": model_version.metadata.tags,
                "parameters": model_version.metadata.parameters,
                "metrics": model_version.metadata.metrics,
                "dependencies": model_version.metadata.dependencies,
                "size_bytes": model_version.metadata.size_bytes,
                "checksum": model_version.metadata.checksum
            },
            "model_path": model_version.model_path,
            "created_at": model_version.created_at.isoformat(),
            "updated_at": model_version.updated_at.isoformat(),
            "deployment_info": model_version.deployment_info
        }
        
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_registry(self):
        """Load existing models from disk"""
        if not os.path.exists(self.registry_path):
            return
        
        for filename in os.listdir(self.registry_path):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.registry_path, filename), 'r') as f:
                        data = json.load(f)
                    
                    # Reconstruct model version
                    metadata = ModelMetadata(
                        name=data["metadata"]["name"],
                        version=data["metadata"]["version"],
                        description=data["metadata"]["description"],
                        model_type=data["metadata"]["model_type"],
                        framework=data["metadata"]["framework"],
                        created_at=datetime.fromisoformat(data["metadata"]["created_at"]),
                        created_by=data["metadata"]["created_by"],
                        tags=data["metadata"]["tags"],
                        parameters=data["metadata"]["parameters"],
                        metrics=data["metadata"]["metrics"],
                        dependencies=data["metadata"]["dependencies"],
                        size_bytes=data["metadata"]["size_bytes"],
                        checksum=data["metadata"]["checksum"]
                    )
                    
                    model_version = ModelVersion(
                        name=data["name"],
                        version=data["version"],
                        status=ModelStatus(data["status"]),
                        stage=ModelStage(data["stage"]),
                        metadata=metadata,
                        model_path=data["model_path"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        updated_at=datetime.fromisoformat(data["updated_at"]),
                        deployment_info=data.get("deployment_info", {})
                    )
                    
                    # Store in memory
                    if model_version.name not in self.models:
                        self.models[model_version.name] = {}
                    self.models[model_version.name][model_version.version] = model_version
                    
                    # Track current stage
                    if model_version.stage != ModelStage.NONE:
                        self.current_stages[model_version.name] = model_version.stage
                
                except Exception as e:
                    logger.error(f"Failed to load model from {filename}: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get registry health status"""
        total_models = sum(len(versions) for versions in self.models.values())
        production_models = sum(1 for versions in self.models.values() 
                              for model in versions.values() 
                              if model.stage == ModelStage.PRODUCTION)
        
        return {
            "status": "healthy",
            "total_models": total_models,
            "unique_model_names": len(self.models),
            "production_models": production_models,
            "registry_path": self.registry_path,
            "last_check": datetime.utcnow().isoformat()
        }
