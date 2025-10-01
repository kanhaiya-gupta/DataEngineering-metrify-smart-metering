"""
Real-time Anomaly Scorer
Real-time anomaly scoring for streaming data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from collections import deque
import threading
import time

logger = logging.getLogger(__name__)

class RealTimeAnomalyScorer:
    """
    Real-time anomaly scoring for streaming smart meter data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.window_size = self.config.get('window_size', 100)
        self.score_threshold = self.config.get('score_threshold', 0.5)
        self.update_interval = self.config.get('update_interval', 60)  # seconds
        
        # Data buffers
        self.data_buffer = deque(maxlen=self.window_size)
        self.score_buffer = deque(maxlen=self.window_size)
        
        # Statistics for real-time updates
        self.running_stats = {
            'mean': 0.0,
            'std': 1.0,
            'min': float('inf'),
            'max': float('-inf'),
            'count': 0
        }
        
        # Anomaly detection state
        self.is_anomaly = False
        self.anomaly_score = 0.0
        self.confidence = 0.0
        
        # Threading
        self.lock = threading.Lock()
        self.running = False
        self.update_thread = None
        
        logger.info("RealTimeAnomalyScorer initialized")
    
    def start(self):
        """Start real-time scoring"""
        if self.running:
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info("Real-time anomaly scoring started")
    
    def stop(self):
        """Stop real-time scoring"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
        
        logger.info("Real-time anomaly scoring stopped")
    
    def add_data_point(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add new data point and calculate anomaly score"""
        with self.lock:
            # Add to buffer
            timestamp = datetime.now()
            data_point = {
                'timestamp': timestamp,
                'data': data
            }
            self.data_buffer.append(data_point)
            
            # Calculate anomaly score
            score = self._calculate_anomaly_score(data)
            self.score_buffer.append(score)
            
            # Update running statistics
            self._update_running_stats(score)
            
            # Determine if anomaly
            self.is_anomaly = score > self.score_threshold
            self.anomaly_score = score
            self.confidence = self._calculate_confidence(score)
            
            return {
                'timestamp': timestamp.isoformat(),
                'anomaly_score': score,
                'is_anomaly': self.is_anomaly,
                'confidence': self.confidence,
                'running_stats': self.running_stats.copy()
            }
    
    def _calculate_anomaly_score(self, data: Dict[str, Any]) -> float:
        """Calculate anomaly score for data point"""
        if len(self.data_buffer) < 2:
            return 0.0
        
        # Extract numeric values
        numeric_values = [v for v in data.values() if isinstance(v, (int, float))]
        
        if not numeric_values:
            return 0.0
        
        # Calculate Z-score based on running statistics
        scores = []
        for value in numeric_values:
            if self.running_stats['std'] > 0:
                z_score = abs(value - self.running_stats['mean']) / self.running_stats['std']
                scores.append(z_score)
        
        # Return maximum Z-score as anomaly score
        return max(scores) if scores else 0.0
    
    def _update_running_stats(self, score: float):
        """Update running statistics"""
        count = self.running_stats['count']
        
        if count == 0:
            self.running_stats['mean'] = score
            self.running_stats['min'] = score
            self.running_stats['max'] = score
        else:
            # Update mean using Welford's algorithm
            old_mean = self.running_stats['mean']
            new_mean = old_mean + (score - old_mean) / (count + 1)
            self.running_stats['mean'] = new_mean
            
            # Update min/max
            self.running_stats['min'] = min(self.running_stats['min'], score)
            self.running_stats['max'] = max(self.running_stats['max'], score)
        
        # Update standard deviation (simplified)
        if count > 0:
            variance = ((count - 1) * self.running_stats['std']**2 + (score - old_mean) * (score - new_mean)) / count
            self.running_stats['std'] = np.sqrt(max(variance, 0))
        
        self.running_stats['count'] += 1
    
    def _calculate_confidence(self, score: float) -> float:
        """Calculate confidence in anomaly detection"""
        if self.running_stats['std'] == 0:
            return 0.0
        
        # Confidence based on how far the score is from the mean
        z_score = abs(score - self.running_stats['mean']) / self.running_stats['std']
        
        # Convert to 0-1 confidence score
        confidence = min(z_score / 3.0, 1.0)  # 3-sigma rule
        
        return confidence
    
    def _update_loop(self):
        """Background update loop"""
        while self.running:
            try:
                self._update_models()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in update loop: {str(e)}")
                time.sleep(1)
    
    def _update_models(self):
        """Update models with recent data"""
        if len(self.data_buffer) < 10:
            return
        
        # This would update any ML models with recent data
        # For now, just log the update
        logger.debug(f"Updated models with {len(self.data_buffer)} data points")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current anomaly detection status"""
        with self.lock:
            return {
                'is_anomaly': self.is_anomaly,
                'anomaly_score': self.anomaly_score,
                'confidence': self.confidence,
                'running_stats': self.running_stats.copy(),
                'buffer_size': len(self.data_buffer),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_recent_scores(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent anomaly scores"""
        with self.lock:
            recent_scores = []
            for i, (data_point, score) in enumerate(zip(list(self.data_buffer)[-n:], list(self.score_buffer)[-n:])):
                recent_scores.append({
                    'timestamp': data_point['timestamp'].isoformat(),
                    'anomaly_score': score,
                    'data': data_point['data']
                })
            
            return recent_scores
    
    def get_anomaly_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get anomaly history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            history = []
            for data_point, score in zip(self.data_buffer, self.score_buffer):
                if data_point['timestamp'] >= cutoff_time:
                    history.append({
                        'timestamp': data_point['timestamp'].isoformat(),
                        'anomaly_score': score,
                        'is_anomaly': score > self.score_threshold,
                        'data': data_point['data']
                    })
            
            return history
    
    def reset(self):
        """Reset the scorer"""
        with self.lock:
            self.data_buffer.clear()
            self.score_buffer.clear()
            self.running_stats = {
                'mean': 0.0,
                'std': 1.0,
                'min': float('inf'),
                'max': float('-inf'),
                'count': 0
            }
            self.is_anomaly = False
            self.anomaly_score = 0.0
            self.confidence = 0.0
        
        logger.info("Real-time anomaly scorer reset")
