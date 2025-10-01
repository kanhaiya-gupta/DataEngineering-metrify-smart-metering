"""
Grid Optimization Model

This module implements ML models for grid optimization and energy management,
including load balancing, demand response, and grid stability optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError as MAE

logger = logging.getLogger(__name__)


@dataclass
class GridOptimizationConfig:
    """Configuration for grid optimization model"""
    model_type: str = "multi_output"  # multi_output, ensemble, reinforcement_learning
    sequence_length: int = 24
    forecast_horizon: int = 24
    optimization_targets: List[str] = None
    load_balancing_weight: float = 0.4
    efficiency_weight: float = 0.3
    stability_weight: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    use_attention: bool = True
    use_residual_connections: bool = True


class GridOptimizationModel:
    """
    Advanced grid optimization model for smart meter data
    
    Optimizes:
    - Load balancing across grid operators
    - Energy efficiency and demand response
    - Grid stability and frequency regulation
    - Cost optimization and resource allocation
    """
    
    def __init__(self, config: GridOptimizationConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.is_trained = False
        
        # Set default optimization targets if not provided
        if self.config.optimization_targets is None:
            self.config.optimization_targets = [
                "load_balancing_score",
                "efficiency_score", 
                "stability_score",
                "cost_optimization_score"
            ]
    
    def create_multi_output_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create multi-output model for grid optimization"""
        inputs = layers.Input(shape=input_shape, name='grid_input')
        
        # Shared LSTM layers
        x = layers.LSTM(128, return_sequences=True, dropout=0.2)(inputs)
        x = layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
        x = layers.LSTM(32, return_sequences=False, dropout=0.2)(x)
        
        # Attention mechanism
        if self.config.use_attention:
            # Self-attention
            attention = layers.MultiHeadAttention(
                num_heads=8,
                key_dim=32,
                dropout=0.2
            )(inputs, inputs)
            
            # Add & Norm
            attention = layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
            
            # Global average pooling
            attention_output = layers.GlobalAveragePooling1D()(attention)
            
            # Combine LSTM and attention outputs
            x = layers.Concatenate()([x, attention_output])
        
        # Shared dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Multiple output heads for different optimization targets
        outputs = []
        
        for target in self.config.optimization_targets:
            if target == "load_balancing_score":
                # Load balancing optimization
                output = layers.Dense(16, activation='relu', name=f'{target}_dense')(x)
                output = layers.Dropout(0.1)(output)
                output = layers.Dense(1, activation='sigmoid', name=target)(output)
                outputs.append(output)
                
            elif target == "efficiency_score":
                # Energy efficiency optimization
                output = layers.Dense(16, activation='relu', name=f'{target}_dense')(x)
                output = layers.Dropout(0.1)(output)
                output = layers.Dense(1, activation='sigmoid', name=target)(output)
                outputs.append(output)
                
            elif target == "stability_score":
                # Grid stability optimization
                output = layers.Dense(16, activation='relu', name=f'{target}_dense')(x)
                output = layers.Dropout(0.1)(output)
                output = layers.Dense(1, activation='sigmoid', name=target)(output)
                outputs.append(output)
                
            elif target == "cost_optimization_score":
                # Cost optimization
                output = layers.Dense(16, activation='relu', name=f'{target}_dense')(x)
                output = layers.Dropout(0.1)(output)
                output = layers.Dense(1, activation='sigmoid', name=target)(output)
                outputs.append(output)
        
        model = keras.Model(inputs, outputs, name='grid_optimization_model')
        return model
    
    def create_ensemble_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create ensemble model combining multiple optimization strategies"""
        inputs = layers.Input(shape=input_shape, name='grid_input')
        
        # Multiple parallel branches
        branches = []
        
        # Branch 1: Load balancing focus
        branch1 = layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
        branch1 = layers.LSTM(32, return_sequences=False, dropout=0.2)(branch1)
        branch1 = layers.Dense(16, activation='relu')(branch1)
        branch1 = layers.Dropout(0.2)(branch1)
        branch1 = layers.Dense(1, activation='sigmoid', name='load_balancing_branch')(branch1)
        branches.append(branch1)
        
        # Branch 2: Efficiency focus
        branch2 = layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
        branch2 = layers.LSTM(32, return_sequences=False, dropout=0.2)(branch2)
        branch2 = layers.Dense(16, activation='relu')(branch2)
        branch2 = layers.Dropout(0.2)(branch2)
        branch2 = layers.Dense(1, activation='sigmoid', name='efficiency_branch')(branch2)
        branches.append(branch2)
        
        # Branch 3: Stability focus
        branch3 = layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
        branch3 = layers.LSTM(32, return_sequences=False, dropout=0.2)(branch3)
        branch3 = layers.Dense(16, activation='relu')(branch3)
        branch3 = layers.Dropout(0.2)(branch3)
        branch3 = layers.Dense(1, activation='sigmoid', name='stability_branch')(branch3)
        branches.append(branch3)
        
        # Branch 4: Cost optimization focus
        branch4 = layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
        branch4 = layers.LSTM(32, return_sequences=False, dropout=0.2)(branch4)
        branch4 = layers.Dense(16, activation='relu')(branch4)
        branch4 = layers.Dropout(0.2)(branch4)
        branch4 = layers.Dense(1, activation='sigmoid', name='cost_branch')(branch4)
        branches.append(branch4)
        
        # Combine branches
        combined = layers.Concatenate()(branches)
        
        # Final optimization layer
        optimization = layers.Dense(32, activation='relu')(combined)
        optimization = layers.Dropout(0.2)(optimization)
        optimization = layers.Dense(16, activation='relu')(optimization)
        optimization = layers.Dropout(0.1)(optimization)
        
        # Weighted combination of optimization scores
        weights = layers.Dense(4, activation='softmax', name='optimization_weights')(optimization)
        weighted_output = layers.Multiply()([combined, weights])
        
        # Final optimization score
        final_score = layers.Dense(1, activation='sigmoid', name='overall_optimization_score')(weighted_output)
        
        # Create outputs list
        outputs = branches + [weights, final_score]
        
        model = keras.Model(inputs, outputs, name='ensemble_grid_optimization')
        return model
    
    def create_reinforcement_learning_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create RL-based model for grid optimization"""
        inputs = layers.Input(shape=input_shape, name='grid_state')
        
        # State representation
        x = layers.LSTM(128, return_sequences=True, dropout=0.2)(inputs)
        x = layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
        x = layers.LSTM(32, return_sequences=False, dropout=0.2)(x)
        
        # Value function (state value)
        value = layers.Dense(64, activation='relu')(x)
        value = layers.Dropout(0.2)(value)
        value = layers.Dense(32, activation='relu')(value)
        value = layers.Dropout(0.1)(value)
        value_output = layers.Dense(1, activation='linear', name='state_value')(value)
        
        # Policy function (action probabilities)
        policy = layers.Dense(64, activation='relu')(x)
        policy = layers.Dropout(0.2)(policy)
        policy = layers.Dense(32, activation='relu')(policy)
        policy = layers.Dropout(0.1)(policy)
        
        # Action space: [load_balancing_action, efficiency_action, stability_action, cost_action]
        policy_output = layers.Dense(4, activation='softmax', name='action_probabilities')(policy)
        
        # Q-values for each action
        q_values = layers.Dense(4, activation='linear', name='q_values')(policy)
        
        model = keras.Model(inputs, [value_output, policy_output, q_values], name='rl_grid_optimization')
        return model
    
    def compile_model(self, model: keras.Model) -> keras.Model:
        """Compile the model with appropriate loss functions and metrics"""
        if self.config.model_type == "multi_output":
            # Different loss functions for different outputs
            loss_functions = {}
            loss_weights = {}
            
            for i, target in enumerate(self.config.optimization_targets):
                loss_functions[target] = 'mse'
                
                if target == "load_balancing_score":
                    loss_weights[target] = self.config.load_balancing_weight
                elif target == "efficiency_score":
                    loss_weights[target] = self.config.efficiency_weight
                elif target == "stability_score":
                    loss_weights[target] = self.config.stability_weight
                else:
                    loss_weights[target] = 0.1
            
            model.compile(
                optimizer=Adam(learning_rate=self.config.learning_rate),
                loss=loss_functions,
                loss_weights=loss_weights,
                metrics=['mae']
            )
            
        elif self.config.model_type == "ensemble":
            # Ensemble model with multiple loss functions
            loss_functions = {
                'load_balancing_branch': 'mse',
                'efficiency_branch': 'mse',
                'stability_branch': 'mse',
                'cost_branch': 'mse',
                'optimization_weights': 'categorical_crossentropy',
                'overall_optimization_score': 'mse'
            }
            
            loss_weights = {
                'load_balancing_branch': 0.2,
                'efficiency_branch': 0.2,
                'stability_branch': 0.2,
                'cost_branch': 0.2,
                'optimization_weights': 0.1,
                'overall_optimization_score': 0.1
            }
            
            model.compile(
                optimizer=Adam(learning_rate=self.config.learning_rate),
                loss=loss_functions,
                loss_weights=loss_weights,
                metrics=['mae']
            )
            
        elif self.config.model_type == "reinforcement_learning":
            # RL model with actor-critic loss
            model.compile(
                optimizer=Adam(learning_rate=self.config.learning_rate),
                loss=['mse', 'categorical_crossentropy', 'mse'],
                loss_weights=[0.3, 0.4, 0.3],
                metrics=['mae']
            )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: Dict[str, np.ndarray],
              X_val: np.ndarray, y_val: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Train the grid optimization model"""
        logger.info(f"Training grid optimization model: {self.config.model_type}")
        
        # Create model
        input_shape = X_train.shape[1:]
        
        if self.config.model_type == "multi_output":
            model = self.create_multi_output_model(input_shape)
        elif self.config.model_type == "ensemble":
            model = self.create_ensemble_model(input_shape)
        elif self.config.model_type == "reinforcement_learning":
            model = self.create_reinforcement_learning_model(input_shape)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        # Compile model
        model = self.compile_model(model)
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath='best_grid_optimization_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Store model
        self.model = model
        self.is_trained = True
        
        logger.info("Grid optimization model training completed")
        
        return {
            "model": model,
            "history": history.history,
            "config": self.config
        }
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make grid optimization predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X, verbose=0)
        
        if self.config.model_type == "multi_output":
            # Return predictions for each optimization target
            result = {}
            for i, target in enumerate(self.config.optimization_targets):
                result[target] = predictions[i]
            return result
            
        elif self.config.model_type == "ensemble":
            # Return ensemble predictions
            return {
                "load_balancing_score": predictions[0],
                "efficiency_score": predictions[1],
                "stability_score": predictions[2],
                "cost_optimization_score": predictions[3],
                "optimization_weights": predictions[4],
                "overall_optimization_score": predictions[5]
            }
            
        elif self.config.model_type == "reinforcement_learning":
            # Return RL predictions
            return {
                "state_value": predictions[0],
                "action_probabilities": predictions[1],
                "q_values": predictions[2]
            }
    
    def optimize_grid(self, X: np.ndarray) -> Dict[str, Any]:
        """Generate grid optimization recommendations"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.predict(X)
        
        # Generate optimization recommendations
        recommendations = []
        
        if self.config.model_type in ["multi_output", "ensemble"]:
            # Analyze optimization scores
            for target, scores in predictions.items():
                if target.endswith("_score") and not target.startswith("overall"):
                    avg_score = np.mean(scores)
                    
                    if avg_score < 0.3:
                        recommendations.append({
                            "type": target.replace("_score", ""),
                            "priority": "high",
                            "score": float(avg_score),
                            "message": f"Low {target.replace('_score', '')} detected. Immediate action required.",
                            "action": self._get_optimization_action(target.replace("_score", ""))
                        })
                    elif avg_score < 0.6:
                        recommendations.append({
                            "type": target.replace("_score", ""),
                            "priority": "medium",
                            "score": float(avg_score),
                            "message": f"Moderate {target.replace('_score', '')}. Consider optimization.",
                            "action": self._get_optimization_action(target.replace("_score", ""))
                        })
        
        elif self.config.model_type == "reinforcement_learning":
            # Analyze RL predictions
            action_probs = predictions["action_probabilities"]
            q_values = predictions["q_values"]
            
            # Get best action
            best_action_idx = np.argmax(action_probs, axis=1)
            action_names = ["load_balancing", "efficiency", "stability", "cost_optimization"]
            
            for i, action_idx in enumerate(best_action_idx):
                action_name = action_names[action_idx]
                confidence = action_probs[i][action_idx]
                q_value = q_values[i][action_idx]
                
                recommendations.append({
                    "type": action_name,
                    "priority": "high" if confidence > 0.7 else "medium",
                    "confidence": float(confidence),
                    "q_value": float(q_value),
                    "message": f"Recommended action: {action_name}",
                    "action": self._get_optimization_action(action_name)
                })
        
        # Calculate overall optimization score
        if "overall_optimization_score" in predictions:
            overall_score = np.mean(predictions["overall_optimization_score"])
        else:
            # Calculate from individual scores
            scores = [np.mean(pred) for pred in predictions.values() if isinstance(pred, np.ndarray)]
            overall_score = np.mean(scores) if scores else 0.0
        
        return {
            "recommendations": recommendations,
            "overall_score": float(overall_score),
            "timestamp": datetime.now().isoformat(),
            "model_type": self.config.model_type
        }
    
    def _get_optimization_action(self, optimization_type: str) -> str:
        """Get specific action for optimization type"""
        actions = {
            "load_balancing": "Redistribute load across grid operators and activate backup resources",
            "efficiency": "Implement demand response programs and optimize energy distribution",
            "stability": "Adjust frequency regulation and voltage control systems",
            "cost_optimization": "Optimize energy procurement and reduce operational costs"
        }
        return actions.get(optimization_type, "Review grid operations and implement improvements")
    
    def evaluate(self, X_test: np.ndarray, y_test: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Evaluate the model on test data"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        predictions = self.predict(X_test)
        
        # Calculate metrics for each optimization target
        metrics = {}
        
        for target in self.config.optimization_targets:
            if target in predictions and target in y_test:
                y_true = y_test[target]
                y_pred = predictions[target]
                
                # Calculate regression metrics
                mse = np.mean((y_true - y_pred) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_true - y_pred))
                
                # Calculate R²
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-8))
                
                metrics[target] = {
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "r2": float(r2)
                }
        
        logger.info("Grid optimization model evaluation completed")
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
