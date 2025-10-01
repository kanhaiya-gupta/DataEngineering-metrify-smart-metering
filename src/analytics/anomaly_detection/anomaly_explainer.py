"""
Anomaly Explainer
Explains and visualizes anomaly detection results
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json

logger = logging.getLogger(__name__)

class AnomalyExplainer:
    """
    Explains and visualizes anomaly detection results
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.explanations = []
        
        logger.info("AnomalyExplainer initialized")
    
    def explain_anomaly(self, 
                       anomaly_data: Dict[str, Any],
                       reference_data: pd.DataFrame,
                       feature_names: List[str]) -> Dict[str, Any]:
        """Explain why a data point is considered anomalous"""
        
        explanation = {
            'anomaly_id': anomaly_data.get('id', 'unknown'),
            'timestamp': anomaly_data.get('timestamp', datetime.now().isoformat()),
            'anomaly_score': anomaly_data.get('anomaly_score', 0.0),
            'confidence': anomaly_data.get('confidence', 0.0),
            'explanations': []
        }
        
        # Feature-based explanations
        feature_explanations = self._explain_features(anomaly_data, reference_data, feature_names)
        explanation['explanations'].extend(feature_explanations)
        
        # Statistical explanations
        statistical_explanations = self._explain_statistics(anomaly_data, reference_data)
        explanation['explanations'].extend(statistical_explanations)
        
        # Temporal explanations
        temporal_explanations = self._explain_temporal_patterns(anomaly_data, reference_data)
        explanation['explanations'].extend(temporal_explanations)
        
        # Overall explanation summary
        explanation['summary'] = self._generate_summary(explanation['explanations'])
        
        # Store explanation
        self.explanations.append(explanation)
        
        return explanation
    
    def _explain_features(self, 
                         anomaly_data: Dict[str, Any],
                         reference_data: pd.DataFrame,
                         feature_names: List[str]) -> List[Dict[str, Any]]:
        """Explain feature-based anomalies"""
        explanations = []
        
        if 'features' not in anomaly_data:
            return explanations
        
        anomaly_features = anomaly_data['features']
        
        for feature_name in feature_names:
            if feature_name not in anomaly_features or feature_name not in reference_data.columns:
                continue
            
            anomaly_value = anomaly_features[feature_name]
            reference_values = reference_data[feature_name].dropna()
            
            if len(reference_values) == 0:
                continue
            
            # Calculate statistics
            ref_mean = reference_values.mean()
            ref_std = reference_values.std()
            ref_min = reference_values.min()
            ref_max = reference_values.max()
            
            # Calculate Z-score
            if ref_std > 0:
                z_score = abs(anomaly_value - ref_mean) / ref_std
            else:
                z_score = 0
            
            # Determine anomaly type
            if anomaly_value > ref_max:
                anomaly_type = "above_maximum"
                severity = "extreme"
            elif anomaly_value < ref_min:
                anomaly_type = "below_minimum"
                severity = "extreme"
            elif z_score > 3:
                anomaly_type = "statistical_outlier"
                severity = "high"
            elif z_score > 2:
                anomaly_type = "moderate_deviation"
                severity = "medium"
            else:
                continue  # Not anomalous for this feature
            
            # Calculate percentile
            percentile = (reference_values < anomaly_value).mean() * 100
            
            explanations.append({
                'type': 'feature_anomaly',
                'feature_name': feature_name,
                'anomaly_type': anomaly_type,
                'severity': severity,
                'anomaly_value': anomaly_value,
                'reference_mean': ref_mean,
                'reference_std': ref_std,
                'reference_min': ref_min,
                'reference_max': ref_max,
                'z_score': z_score,
                'percentile': percentile,
                'explanation': f"Feature '{feature_name}' has value {anomaly_value:.3f}, which is {anomaly_type} compared to reference data (mean: {ref_mean:.3f}, std: {ref_std:.3f})"
            })
        
        return explanations
    
    def _explain_statistics(self, 
                           anomaly_data: Dict[str, Any],
                           reference_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Explain statistical anomalies"""
        explanations = []
        
        if 'features' not in anomaly_data:
            return explanations
        
        anomaly_features = anomaly_data['features']
        
        # Get numeric features
        numeric_features = [f for f in anomaly_features.keys() 
                          if isinstance(anomaly_features[f], (int, float)) and f in reference_data.columns]
        
        if len(numeric_features) < 2:
            return explanations
        
        # Calculate multivariate statistics
        anomaly_vector = np.array([anomaly_features[f] for f in numeric_features])
        reference_matrix = reference_data[numeric_features].dropna().values
        
        if len(reference_matrix) < 2:
            return explanations
        
        # Calculate Mahalanobis distance
        ref_mean = np.mean(reference_matrix, axis=0)
        ref_cov = np.cov(reference_matrix.T)
        
        try:
            inv_cov = np.linalg.inv(ref_cov)
            mahal_distance = np.sqrt((anomaly_vector - ref_mean).T @ inv_cov @ (anomaly_vector - ref_mean))
            
            # Calculate threshold (chi-squared distribution)
            threshold = np.sqrt(2 * len(numeric_features))  # Simplified threshold
            
            if mahal_distance > threshold:
                explanations.append({
                    'type': 'multivariate_anomaly',
                    'mahalanobis_distance': mahal_distance,
                    'threshold': threshold,
                    'severity': 'high' if mahal_distance > 2 * threshold else 'medium',
                    'explanation': f"Multivariate anomaly detected with Mahalanobis distance {mahal_distance:.3f} (threshold: {threshold:.3f})"
                })
        except np.linalg.LinAlgError:
            # Singular matrix, use simplified approach
            pass
        
        # Calculate correlation anomalies
        for i, feature1 in enumerate(numeric_features):
            for j, feature2 in enumerate(numeric_features[i+1:], i+1):
                if feature1 in reference_data.columns and feature2 in reference_data.columns:
                    ref_corr = reference_data[feature1].corr(reference_data[feature2])
                    
                    if not np.isnan(ref_corr):
                        # Calculate expected value based on correlation
                        ref1_mean = reference_data[feature1].mean()
                        ref1_std = reference_data[feature1].std()
                        ref2_mean = reference_data[feature2].mean()
                        ref2_std = reference_data[feature2].std()
                        
                        if ref1_std > 0 and ref2_std > 0:
                            expected_value2 = ref2_mean + ref_corr * (ref2_std / ref1_std) * (anomaly_features[feature1] - ref1_mean)
                            actual_value2 = anomaly_features[feature2]
                            
                            deviation = abs(actual_value2 - expected_value2) / ref2_std
                            
                            if deviation > 2:  # 2 standard deviations
                                explanations.append({
                                    'type': 'correlation_anomaly',
                                    'feature1': feature1,
                                    'feature2': feature2,
                                    'correlation': ref_corr,
                                    'expected_value': expected_value2,
                                    'actual_value': actual_value2,
                                    'deviation': deviation,
                                    'severity': 'high' if deviation > 3 else 'medium',
                                    'explanation': f"Correlation anomaly between '{feature1}' and '{feature2}': expected {expected_value2:.3f}, got {actual_value2:.3f} (deviation: {deviation:.3f}σ)"
                                })
        
        return explanations
    
    def _explain_temporal_patterns(self, 
                                  anomaly_data: Dict[str, Any],
                                  reference_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Explain temporal pattern anomalies"""
        explanations = []
        
        if not isinstance(reference_data.index, pd.DatetimeIndex):
            return explanations
        
        anomaly_time = pd.to_datetime(anomaly_data.get('timestamp', datetime.now()))
        
        # Hour of day anomaly
        anomaly_hour = anomaly_time.hour
        reference_hours = reference_data.index.hour
        
        # Calculate typical values for this hour
        hour_data = reference_data[reference_data.index.hour == anomaly_hour]
        
        if len(hour_data) > 0:
            # Check if consumption is anomalous for this hour
            if 'consumption' in hour_data.columns:
                hour_consumption = hour_data['consumption'].dropna()
                if len(hour_consumption) > 0:
                    hour_mean = hour_consumption.mean()
                    hour_std = hour_consumption.std()
                    
                    if 'features' in anomaly_data and 'consumption' in anomaly_data['features']:
                        anomaly_consumption = anomaly_data['features']['consumption']
                        
                        if hour_std > 0:
                            z_score = abs(anomaly_consumption - hour_mean) / hour_std
                            
                            if z_score > 2:
                                explanations.append({
                                    'type': 'temporal_anomaly',
                                    'pattern': 'hourly_deviation',
                                    'hour': anomaly_hour,
                                    'anomaly_value': anomaly_consumption,
                                    'typical_value': hour_mean,
                                    'z_score': z_score,
                                    'severity': 'high' if z_score > 3 else 'medium',
                                    'explanation': f"Consumption at hour {anomaly_hour} is {z_score:.2f}σ away from typical value ({hour_mean:.3f})"
                                })
        
        # Day of week anomaly
        anomaly_dow = anomaly_time.dayofweek
        dow_data = reference_data[reference_data.index.dayofweek == anomaly_dow]
        
        if len(dow_data) > 0 and 'consumption' in dow_data.columns:
            dow_consumption = dow_data['consumption'].dropna()
            if len(dow_consumption) > 0:
                dow_mean = dow_consumption.mean()
                dow_std = dow_consumption.std()
                
                if 'features' in anomaly_data and 'consumption' in anomaly_data['features']:
                    anomaly_consumption = anomaly_data['features']['consumption']
                    
                    if dow_std > 0:
                        z_score = abs(anomaly_consumption - dow_mean) / dow_std
                        
                        if z_score > 2:
                            day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][anomaly_dow]
                            explanations.append({
                                'type': 'temporal_anomaly',
                                'pattern': 'weekly_deviation',
                                'day_of_week': day_name,
                                'anomaly_value': anomaly_consumption,
                                'typical_value': dow_mean,
                                'z_score': z_score,
                                'severity': 'high' if z_score > 3 else 'medium',
                                'explanation': f"Consumption on {day_name} is {z_score:.2f}σ away from typical value ({dow_mean:.3f})"
                            })
        
        return explanations
    
    def _generate_summary(self, explanations: List[Dict[str, Any]]) -> str:
        """Generate a summary of all explanations"""
        if not explanations:
            return "No specific explanations available for this anomaly."
        
        # Count by type
        type_counts = {}
        severity_counts = {}
        
        for exp in explanations:
            exp_type = exp.get('type', 'unknown')
            severity = exp.get('severity', 'unknown')
            
            type_counts[exp_type] = type_counts.get(exp_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Generate summary
        summary_parts = []
        
        if 'feature_anomaly' in type_counts:
            summary_parts.append(f"{type_counts['feature_anomaly']} feature(s) show anomalous values")
        
        if 'multivariate_anomaly' in type_counts:
            summary_parts.append("multivariate statistical anomaly detected")
        
        if 'correlation_anomaly' in type_counts:
            summary_parts.append(f"{type_counts['correlation_anomaly']} correlation anomaly(ies) found")
        
        if 'temporal_anomaly' in type_counts:
            summary_parts.append("temporal pattern anomaly detected")
        
        # Add severity information
        if 'extreme' in severity_counts:
            summary_parts.append(f"({severity_counts['extreme']} extreme, {severity_counts.get('high', 0)} high, {severity_counts.get('medium', 0)} medium severity)")
        elif 'high' in severity_counts:
            summary_parts.append(f"({severity_counts['high']} high, {severity_counts.get('medium', 0)} medium severity)")
        elif 'medium' in severity_counts:
            summary_parts.append(f"({severity_counts['medium']} medium severity)")
        
        return "Anomaly detected: " + ", ".join(summary_parts) + "."
    
    def visualize_anomaly(self, 
                         anomaly_data: Dict[str, Any],
                         reference_data: pd.DataFrame,
                         feature_names: List[str],
                         save_path: Optional[str] = None) -> Dict[str, Any]:
        """Create visualizations for anomaly explanation"""
        
        if 'features' not in anomaly_data:
            return {"error": "No feature data available for visualization"}
        
        anomaly_features = anomaly_data['features']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Anomaly Explanation - {anomaly_data.get('timestamp', 'Unknown')}", fontsize=16)
        
        # 1. Feature comparison
        ax1 = axes[0, 0]
        available_features = [f for f in feature_names if f in anomaly_features and f in reference_data.columns]
        
        if available_features:
            anomaly_values = [anomaly_features[f] for f in available_features]
            ref_means = [reference_data[f].mean() for f in available_features]
            ref_stds = [reference_data[f].std() for f in available_features]
            
            x_pos = np.arange(len(available_features))
            width = 0.35
            
            ax1.bar(x_pos - width/2, anomaly_values, width, label='Anomaly', alpha=0.7, color='red')
            ax1.bar(x_pos + width/2, ref_means, width, label='Reference Mean', alpha=0.7, color='blue')
            ax1.errorbar(x_pos + width/2, ref_means, yerr=ref_stds, fmt='none', color='black', capsize=5)
            
            ax1.set_xlabel('Features')
            ax1.set_ylabel('Values')
            ax1.set_title('Feature Comparison')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(available_features, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Z-scores
        ax2 = axes[0, 1]
        z_scores = []
        for f in available_features:
            if f in reference_data.columns:
                ref_mean = reference_data[f].mean()
                ref_std = reference_data[f].std()
                if ref_std > 0:
                    z_score = abs(anomaly_features[f] - ref_mean) / ref_std
                    z_scores.append(z_score)
                else:
                    z_scores.append(0)
        
        if z_scores:
            colors = ['red' if z > 2 else 'orange' if z > 1 else 'green' for z in z_scores]
            ax2.bar(range(len(z_scores)), z_scores, color=colors, alpha=0.7)
            ax2.axhline(y=2, color='red', linestyle='--', label='2σ threshold')
            ax2.axhline(y=3, color='darkred', linestyle='--', label='3σ threshold')
            ax2.set_xlabel('Features')
            ax2.set_ylabel('Z-Score')
            ax2.set_title('Feature Z-Scores')
            ax2.set_xticks(range(len(available_features)))
            ax2.set_xticklabels(available_features, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Time series (if available)
        ax3 = axes[1, 0]
        if 'consumption' in reference_data.columns and isinstance(reference_data.index, pd.DatetimeIndex):
            # Plot recent data
            recent_data = reference_data.tail(100)
            ax3.plot(recent_data.index, recent_data['consumption'], label='Recent Data', alpha=0.7)
            
            # Mark anomaly point
            anomaly_time = pd.to_datetime(anomaly_data.get('timestamp', datetime.now()))
            if 'consumption' in anomaly_features:
                ax3.scatter(anomaly_time, anomaly_features['consumption'], 
                           color='red', s=100, label='Anomaly', zorder=5)
            
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Consumption')
            ax3.set_title('Time Series Context')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Distribution comparison
        ax4 = axes[1, 1]
        if 'consumption' in reference_data.columns and 'consumption' in anomaly_features:
            ref_consumption = reference_data['consumption'].dropna()
            ax4.hist(ref_consumption, bins=30, alpha=0.7, label='Reference Distribution', density=True)
            ax4.axvline(anomaly_features['consumption'], color='red', linestyle='--', 
                       linewidth=2, label='Anomaly Value')
            ax4.set_xlabel('Consumption')
            ax4.set_ylabel('Density')
            ax4.set_title('Distribution Comparison')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return {"visualization_saved": save_path}
        else:
            plt.show()
            return {"visualization_displayed": True}
    
    def export_explanations(self, filepath: str):
        """Export all explanations to file"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_explanations': len(self.explanations),
            'explanations': self.explanations
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Anomaly explanations exported to {filepath}")
    
    def get_explanation_summary(self) -> Dict[str, Any]:
        """Get summary of all explanations"""
        if not self.explanations:
            return {"message": "No explanations available"}
        
        # Aggregate statistics
        total_anomalies = len(self.explanations)
        
        # Count by type
        type_counts = {}
        severity_counts = {}
        
        for exp in self.explanations:
            for explanation in exp.get('explanations', []):
                exp_type = explanation.get('type', 'unknown')
                severity = explanation.get('severity', 'unknown')
                
                type_counts[exp_type] = type_counts.get(exp_type, 0) + 1
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_anomalies_explained': total_anomalies,
            'explanation_type_counts': type_counts,
            'severity_counts': severity_counts,
            'latest_explanation': self.explanations[-1]['timestamp'] if self.explanations else None
        }
