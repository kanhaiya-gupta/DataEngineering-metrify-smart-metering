"""
Risk Assessor
Assesses and quantifies risks in smart meter infrastructure
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskCategory(Enum):
    """Risk category enumeration"""
    EQUIPMENT_FAILURE = "equipment_failure"
    DATA_QUALITY = "data_quality"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"

@dataclass
class RiskFactor:
    """Represents a risk factor"""
    factor_id: str
    category: RiskCategory
    name: str
    description: str
    probability: float  # 0-1
    impact: float  # 0-1
    risk_score: float  # probability * impact
    mitigation_actions: List[str]
    last_updated: datetime

@dataclass
class RiskAssessment:
    """Represents a risk assessment"""
    assessment_id: str
    equipment_id: str
    timestamp: datetime
    overall_risk_score: float
    risk_level: RiskLevel
    risk_factors: List[RiskFactor]
    recommendations: List[str]
    next_assessment_date: datetime

class RiskAssessor:
    """
    Assesses and quantifies risks in smart meter infrastructure
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.risk_factors = []
        self.assessments = []
        self.risk_thresholds = {
            'low': 0.2,
            'medium': 0.4,
            'high': 0.7,
            'critical': 0.9
        }
        
        logger.info("RiskAssessor initialized")
    
    def add_risk_factor(self, risk_factor: RiskFactor):
        """Add a risk factor"""
        self.risk_factors.append(risk_factor)
        logger.info(f"Added risk factor: {risk_factor.factor_id}")
    
    def assess_equipment_risk(self, 
                            equipment_id: str,
                            equipment_data: pd.DataFrame,
                            assessment_date: Optional[datetime] = None) -> RiskAssessment:
        """Assess risk for specific equipment"""
        
        if assessment_date is None:
            assessment_date = datetime.now()
        
        # Calculate risk factors
        risk_factors = self._calculate_equipment_risk_factors(equipment_id, equipment_data, assessment_date)
        
        # Calculate overall risk score
        overall_risk_score = self._calculate_overall_risk_score(risk_factors)
        
        # Determine risk level
        risk_level = self._determine_risk_level(overall_risk_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk_factors, risk_level)
        
        # Create assessment
        assessment = RiskAssessment(
            assessment_id=f"ASSESS_{equipment_id}_{assessment_date.strftime('%Y%m%d_%H%M%S')}",
            equipment_id=equipment_id,
            timestamp=assessment_date,
            overall_risk_score=overall_risk_score,
            risk_level=risk_level,
            risk_factors=risk_factors,
            recommendations=recommendations,
            next_assessment_date=assessment_date + timedelta(days=30)
        )
        
        self.assessments.append(assessment)
        
        logger.info(f"Risk assessment completed for equipment {equipment_id}: {risk_level.value}")
        
        return assessment
    
    def _calculate_equipment_risk_factors(self, 
                                        equipment_id: str,
                                        equipment_data: pd.DataFrame,
                                        assessment_date: datetime) -> List[RiskFactor]:
        """Calculate risk factors for equipment"""
        risk_factors = []
        
        # Equipment age risk
        if 'installation_date' in equipment_data.columns:
            installation_date = pd.to_datetime(equipment_data['installation_date'].iloc[0])
            age_years = (assessment_date - installation_date).days / 365.25
            
            # Age risk increases exponentially after 10 years
            age_probability = min(1.0, max(0.0, (age_years - 5) / 15))
            age_impact = 0.8  # High impact for equipment failure
            
            risk_factors.append(RiskFactor(
                factor_id=f"AGE_{equipment_id}",
                category=RiskCategory.EQUIPMENT_FAILURE,
                name="Equipment Age Risk",
                description=f"Equipment is {age_years:.1f} years old",
                probability=age_probability,
                impact=age_impact,
                risk_score=age_probability * age_impact,
                mitigation_actions=[
                    "Schedule preventive maintenance",
                    "Consider equipment replacement",
                    "Increase monitoring frequency"
                ],
                last_updated=assessment_date
            ))
        
        # Temperature stress risk
        if 'temperature' in equipment_data.columns:
            temp_data = equipment_data['temperature'].dropna()
            if len(temp_data) > 0:
                temp_mean = temp_data.mean()
                temp_std = temp_data.std()
                temp_extremes = ((temp_data < -10) | (temp_data > 40)).sum()
                
                # Temperature stress probability
                temp_probability = min(1.0, temp_extremes / len(temp_data))
                temp_impact = 0.6  # Medium-high impact
                
                risk_factors.append(RiskFactor(
                    factor_id=f"TEMP_{equipment_id}",
                    category=RiskCategory.EQUIPMENT_FAILURE,
                    name="Temperature Stress Risk",
                    description=f"Equipment exposed to extreme temperatures ({temp_extremes} times)",
                    probability=temp_probability,
                    impact=temp_impact,
                    risk_score=temp_probability * temp_impact,
                    mitigation_actions=[
                        "Improve thermal protection",
                        "Relocate equipment if possible",
                        "Install temperature monitoring"
                    ],
                    last_updated=assessment_date
                ))
        
        # Voltage stress risk
        if 'voltage' in equipment_data.columns:
            voltage_data = equipment_data['voltage'].dropna()
            if len(voltage_data) > 0:
                voltage_mean = voltage_data.mean()
                voltage_std = voltage_data.std()
                voltage_deviations = ((voltage_data < 200) | (voltage_data > 250)).sum()
                
                # Voltage stress probability
                voltage_probability = min(1.0, voltage_deviations / len(voltage_data))
                voltage_impact = 0.7  # High impact
                
                risk_factors.append(RiskFactor(
                    factor_id=f"VOLTAGE_{equipment_id}",
                    category=RiskCategory.EQUIPMENT_FAILURE,
                    name="Voltage Stress Risk",
                    description=f"Voltage deviations detected ({voltage_deviations} times)",
                    probability=voltage_probability,
                    impact=voltage_impact,
                    risk_score=voltage_probability * voltage_impact,
                    mitigation_actions=[
                        "Check power supply stability",
                        "Install voltage regulation",
                        "Schedule electrical inspection"
                    ],
                    last_updated=assessment_date
                ))
        
        # Data quality risk
        if 'consumption' in equipment_data.columns:
            consumption_data = equipment_data['consumption'].dropna()
            if len(consumption_data) > 0:
                missing_data = equipment_data['consumption'].isnull().sum()
                data_quality_probability = min(1.0, missing_data / len(equipment_data))
                data_quality_impact = 0.5  # Medium impact
                
                risk_factors.append(RiskFactor(
                    factor_id=f"DATA_QUALITY_{equipment_id}",
                    category=RiskCategory.DATA_QUALITY,
                    name="Data Quality Risk",
                    description=f"Missing data points: {missing_data}",
                    probability=data_quality_probability,
                    impact=data_quality_impact,
                    risk_score=data_quality_probability * data_quality_impact,
                    mitigation_actions=[
                        "Investigate data transmission issues",
                        "Implement data validation",
                        "Set up data quality monitoring"
                    ],
                    last_updated=assessment_date
                ))
        
        # Maintenance history risk
        if 'last_maintenance' in equipment_data.columns:
            last_maintenance = pd.to_datetime(equipment_data['last_maintenance'].iloc[0])
            days_since_maintenance = (assessment_date - last_maintenance).days
            
            # Maintenance risk increases with time
            maintenance_probability = min(1.0, days_since_maintenance / 365)
            maintenance_impact = 0.6  # Medium-high impact
            
            risk_factors.append(RiskFactor(
                factor_id=f"MAINTENANCE_{equipment_id}",
                category=RiskCategory.OPERATIONAL,
                name="Maintenance Overdue Risk",
                description=f"Last maintenance: {days_since_maintenance} days ago",
                probability=maintenance_probability,
                impact=maintenance_impact,
                risk_score=maintenance_probability * maintenance_impact,
                mitigation_actions=[
                    "Schedule immediate maintenance",
                    "Implement predictive maintenance",
                    "Increase inspection frequency"
                ],
                last_updated=assessment_date
            ))
        
        # Failure history risk
        if 'failure_count' in equipment_data.columns:
            failure_count = equipment_data['failure_count'].iloc[0]
            if failure_count > 0:
                failure_probability = min(1.0, failure_count / 10)  # Normalize to 0-1
                failure_impact = 0.9  # Very high impact
                
                risk_factors.append(RiskFactor(
                    factor_id=f"FAILURE_HISTORY_{equipment_id}",
                    category=RiskCategory.EQUIPMENT_FAILURE,
                    name="Failure History Risk",
                    description=f"Previous failures: {failure_count}",
                    probability=failure_probability,
                    impact=failure_impact,
                    risk_score=failure_probability * failure_impact,
                    mitigation_actions=[
                        "Conduct root cause analysis",
                        "Implement enhanced monitoring",
                        "Consider equipment replacement"
                    ],
                    last_updated=assessment_date
                ))
        
        return risk_factors
    
    def _calculate_overall_risk_score(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate overall risk score from individual risk factors"""
        if not risk_factors:
            return 0.0
        
        # Weighted average of risk scores
        total_weight = 0
        weighted_sum = 0
        
        for factor in risk_factors:
            # Weight by impact (higher impact = higher weight)
            weight = factor.impact
            weighted_sum += factor.risk_score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on risk score"""
        if risk_score >= self.risk_thresholds['critical']:
            return RiskLevel.CRITICAL
        elif risk_score >= self.risk_thresholds['high']:
            return RiskLevel.HIGH
        elif risk_score >= self.risk_thresholds['medium']:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_recommendations(self, 
                                risk_factors: List[RiskFactor],
                                risk_level: RiskLevel) -> List[str]:
        """Generate recommendations based on risk factors and level"""
        recommendations = []
        
        # High-level recommendations based on risk level
        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "IMMEDIATE ACTION REQUIRED: Schedule emergency maintenance",
                "Consider temporary equipment shutdown if safety is at risk",
                "Escalate to management and technical team",
                "Implement 24/7 monitoring"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "Schedule maintenance within 7 days",
                "Increase monitoring frequency to daily",
                "Prepare contingency plans",
                "Notify relevant stakeholders"
            ])
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "Schedule maintenance within 30 days",
                "Increase monitoring frequency to weekly",
                "Review maintenance procedures",
                "Consider preventive measures"
            ])
        else:
            recommendations.extend([
                "Continue regular monitoring",
                "Schedule routine maintenance",
                "Maintain current procedures"
            ])
        
        # Specific recommendations based on risk factors
        for factor in risk_factors:
            if factor.risk_score > 0.5:  # High-risk factors
                recommendations.extend(factor.mitigation_actions)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def assess_portfolio_risk(self, 
                            equipment_assessments: List[RiskAssessment]) -> Dict[str, Any]:
        """Assess risk for entire equipment portfolio"""
        
        if not equipment_assessments:
            return {"error": "No equipment assessments available"}
        
        # Aggregate risk metrics
        total_equipment = len(equipment_assessments)
        risk_level_counts = {}
        
        for level in RiskLevel:
            risk_level_counts[level.value] = sum(1 for a in equipment_assessments if a.risk_level == level)
        
        # Calculate portfolio risk score
        portfolio_risk_score = np.mean([a.overall_risk_score for a in equipment_assessments])
        
        # Identify critical equipment
        critical_equipment = [a for a in equipment_assessments if a.risk_level == RiskLevel.CRITICAL]
        high_risk_equipment = [a for a in equipment_assessments if a.risk_level == RiskLevel.HIGH]
        
        # Calculate risk distribution
        risk_distribution = {
            'critical': len(critical_equipment),
            'high': len(high_risk_equipment),
            'medium': risk_level_counts['medium'],
            'low': risk_level_counts['low']
        }
        
        # Generate portfolio recommendations
        portfolio_recommendations = []
        
        if critical_equipment:
            portfolio_recommendations.append(f"URGENT: {len(critical_equipment)} equipment items require immediate attention")
        
        if high_risk_equipment:
            portfolio_recommendations.append(f"PRIORITY: {len(high_risk_equipment)} equipment items need high-priority maintenance")
        
        if portfolio_risk_score > 0.5:
            portfolio_recommendations.append("Consider increasing overall maintenance budget")
            portfolio_recommendations.append("Implement portfolio-wide risk monitoring")
        
        return {
            'portfolio_risk_score': portfolio_risk_score,
            'total_equipment': total_equipment,
            'risk_distribution': risk_distribution,
            'critical_equipment_count': len(critical_equipment),
            'high_risk_equipment_count': len(high_risk_equipment),
            'portfolio_recommendations': portfolio_recommendations,
            'assessment_date': datetime.now().isoformat()
        }
    
    def get_risk_trends(self, 
                       equipment_id: str,
                       days: int = 90) -> Dict[str, Any]:
        """Get risk trends for specific equipment"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        equipment_assessments = [
            a for a in self.assessments 
            if a.equipment_id == equipment_id and a.timestamp >= cutoff_date
        ]
        
        if not equipment_assessments:
            return {"error": f"No assessments found for equipment {equipment_id} in the last {days} days"}
        
        # Sort by timestamp
        equipment_assessments.sort(key=lambda x: x.timestamp)
        
        # Extract trend data
        timestamps = [a.timestamp.isoformat() for a in equipment_assessments]
        risk_scores = [a.overall_risk_score for a in equipment_assessments]
        risk_levels = [a.risk_level.value for a in equipment_assessments]
        
        # Calculate trend
        if len(risk_scores) > 1:
            trend_slope = np.polyfit(range(len(risk_scores)), risk_scores, 1)[0]
            trend_direction = "increasing" if trend_slope > 0.01 else "decreasing" if trend_slope < -0.01 else "stable"
        else:
            trend_direction = "insufficient_data"
        
        return {
            'equipment_id': equipment_id,
            'trend_direction': trend_direction,
            'assessment_count': len(equipment_assessments),
            'current_risk_score': risk_scores[-1] if risk_scores else 0,
            'current_risk_level': risk_levels[-1] if risk_levels else "unknown",
            'timestamps': timestamps,
            'risk_scores': risk_scores,
            'risk_levels': risk_levels
        }
    
    def export_risk_report(self, filepath: str):
        """Export risk assessment report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_assessments': len(self.assessments),
            'total_risk_factors': len(self.risk_factors),
            'assessments': [
                {
                    'assessment_id': a.assessment_id,
                    'equipment_id': a.equipment_id,
                    'timestamp': a.timestamp.isoformat(),
                    'overall_risk_score': a.overall_risk_score,
                    'risk_level': a.risk_level.value,
                    'recommendations': a.recommendations,
                    'risk_factors': [
                        {
                            'factor_id': f.factor_id,
                            'category': f.category.value,
                            'name': f.name,
                            'probability': f.probability,
                            'impact': f.impact,
                            'risk_score': f.risk_score,
                            'mitigation_actions': f.mitigation_actions
                        }
                        for f in a.risk_factors
                    ]
                }
                for a in self.assessments
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Risk assessment report exported to {filepath}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of all risk assessments"""
        if not self.assessments:
            return {"message": "No risk assessments available"}
        
        # Calculate summary statistics
        total_assessments = len(self.assessments)
        risk_level_counts = {}
        
        for level in RiskLevel:
            risk_level_counts[level.value] = sum(1 for a in self.assessments if a.risk_level == level)
        
        # Calculate average risk score
        avg_risk_score = np.mean([a.overall_risk_score for a in self.assessments])
        
        # Get unique equipment
        unique_equipment = len(set(a.equipment_id for a in self.assessments))
        
        return {
            'total_assessments': total_assessments,
            'unique_equipment': unique_equipment,
            'average_risk_score': avg_risk_score,
            'risk_level_distribution': risk_level_counts,
            'latest_assessment': max(self.assessments, key=lambda x: x.timestamp).timestamp.isoformat() if self.assessments else None
        }
