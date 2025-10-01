"""
GDPR Processor
Handles GDPR compliance automation for data processing
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
import uuid

logger = logging.getLogger(__name__)

class GDPRArticle(Enum):
    """GDPR Articles"""
    ARTICLE_6 = "lawful_basis"  # Lawfulness of processing
    ARTICLE_7 = "consent"       # Conditions for consent
    ARTICLE_12 = "transparency"  # Transparent information
    ARTICLE_13 = "info_collection"  # Information to be provided
    ARTICLE_14 = "info_third_party"  # Information from third parties
    ARTICLE_15 = "access"       # Right of access
    ARTICLE_16 = "rectification"  # Right to rectification
    ARTICLE_17 = "erasure"      # Right to erasure
    ARTICLE_18 = "restriction"  # Right to restriction
    ARTICLE_20 = "portability"  # Right to data portability
    ARTICLE_21 = "objection"    # Right to object
    ARTICLE_25 = "privacy_by_design"  # Data protection by design
    ARTICLE_32 = "security"     # Security of processing
    ARTICLE_33 = "breach_notification"  # Breach notification
    ARTICLE_35 = "dpo"          # Data protection impact assessment

@dataclass
class DataSubject:
    """Represents a data subject"""
    subject_id: str
    email: str
    name: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    consent_given: bool = False
    consent_date: Optional[datetime] = None
    data_categories: List[str] = None
    processing_purposes: List[str] = None

@dataclass
class ProcessingActivity:
    """Represents a data processing activity"""
    activity_id: str
    name: str
    purpose: str
    lawful_basis: str
    data_categories: List[str]
    data_subjects: List[str]
    retention_period: int  # in days
    created_at: datetime
    updated_at: datetime

@dataclass
class ConsentRecord:
    """Represents a consent record"""
    consent_id: str
    subject_id: str
    purpose: str
    data_categories: List[str]
    given_at: datetime
    withdrawn_at: Optional[datetime] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class GDPRProcessor:
    """
    Handles GDPR compliance automation for data processing
    """
    
    def __init__(self):
        self.data_subjects = {}
        self.processing_activities = {}
        self.consent_records = {}
        self.breach_records = []
        
        logger.info("GDPRProcessor initialized")
    
    def register_data_subject(self,
                            email: str,
                            name: Optional[str] = None,
                            phone: Optional[str] = None,
                            address: Optional[str] = None) -> str:
        """Register a new data subject"""
        try:
            subject_id = self._generate_subject_id(email)
            
            subject = DataSubject(
                subject_id=subject_id,
                email=email,
                name=name,
                phone=phone,
                address=address,
                data_categories=[],
                processing_purposes=[]
            )
            
            self.data_subjects[subject_id] = subject
            
            logger.info(f"Data subject registered: {subject_id}")
            return subject_id
            
        except Exception as e:
            logger.error(f"Failed to register data subject: {str(e)}")
            return ""
    
    def register_processing_activity(self,
                                   name: str,
                                   purpose: str,
                                   lawful_basis: str,
                                   data_categories: List[str],
                                   data_subjects: List[str],
                                   retention_period: int) -> str:
        """Register a data processing activity"""
        try:
            activity_id = str(uuid.uuid4())
            
            activity = ProcessingActivity(
                activity_id=activity_id,
                name=name,
                purpose=purpose,
                lawful_basis=lawful_basis,
                data_categories=data_categories,
                data_subjects=data_subjects,
                retention_period=retention_period,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.processing_activities[activity_id] = activity
            
            logger.info(f"Processing activity registered: {activity_id}")
            return activity_id
            
        except Exception as e:
            logger.error(f"Failed to register processing activity: {str(e)}")
            return ""
    
    def record_consent(self,
                      subject_id: str,
                      purpose: str,
                      data_categories: List[str],
                      ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None) -> str:
        """Record consent from a data subject"""
        try:
            if subject_id not in self.data_subjects:
                raise ValueError(f"Data subject {subject_id} not found")
            
            consent_id = str(uuid.uuid4())
            
            consent = ConsentRecord(
                consent_id=consent_id,
                subject_id=subject_id,
                purpose=purpose,
                data_categories=data_categories,
                given_at=datetime.now(),
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self.consent_records[consent_id] = consent
            
            # Update data subject
            subject = self.data_subjects[subject_id]
            subject.consent_given = True
            subject.consent_date = datetime.now()
            subject.data_categories.extend(data_categories)
            subject.processing_purposes.append(purpose)
            
            logger.info(f"Consent recorded: {consent_id}")
            return consent_id
            
        except Exception as e:
            logger.error(f"Failed to record consent: {str(e)}")
            return ""
    
    def withdraw_consent(self, consent_id: str) -> bool:
        """Withdraw consent"""
        try:
            if consent_id not in self.consent_records:
                raise ValueError(f"Consent {consent_id} not found")
            
            consent = self.consent_records[consent_id]
            consent.withdrawn_at = datetime.now()
            
            # Update data subject
            subject_id = consent.subject_id
            if subject_id in self.data_subjects:
                subject = self.data_subjects[subject_id]
                subject.consent_given = False
                subject.consent_date = None
            
            logger.info(f"Consent withdrawn: {consent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to withdraw consent: {str(e)}")
            return False
    
    def process_data_subject_request(self,
                                   subject_id: str,
                                   request_type: str,
                                   request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data subject rights requests"""
        try:
            if subject_id not in self.data_subjects:
                raise ValueError(f"Data subject {subject_id} not found")
            
            subject = self.data_subjects[subject_id]
            
            if request_type == "access":
                return self._process_access_request(subject, request_data)
            elif request_type == "rectification":
                return self._process_rectification_request(subject, request_data)
            elif request_type == "erasure":
                return self._process_erasure_request(subject, request_data)
            elif request_type == "portability":
                return self._process_portability_request(subject, request_data)
            elif request_type == "objection":
                return self._process_objection_request(subject, request_data)
            else:
                raise ValueError(f"Unknown request type: {request_type}")
                
        except Exception as e:
            logger.error(f"Failed to process data subject request: {str(e)}")
            return {"error": str(e)}
    
    def _process_access_request(self, subject: DataSubject, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Article 15 - Right of access"""
        try:
            # Collect all data about the subject
            subject_data = {
                "personal_data": {
                    "subject_id": subject.subject_id,
                    "email": subject.email,
                    "name": subject.name,
                    "phone": subject.phone,
                    "address": subject.address
                },
                "processing_activities": [],
                "consent_records": [],
                "data_categories": subject.data_categories,
                "processing_purposes": subject.processing_purposes
            }
            
            # Find relevant processing activities
            for activity in self.processing_activities.values():
                if subject.subject_id in activity.data_subjects:
                    subject_data["processing_activities"].append({
                        "activity_id": activity.activity_id,
                        "name": activity.name,
                        "purpose": activity.purpose,
                        "lawful_basis": activity.lawful_basis,
                        "data_categories": activity.data_categories,
                        "retention_period": activity.retention_period
                    })
            
            # Find consent records
            for consent in self.consent_records.values():
                if consent.subject_id == subject.subject_id:
                    subject_data["consent_records"].append({
                        "consent_id": consent.consent_id,
                        "purpose": consent.purpose,
                        "data_categories": consent.data_categories,
                        "given_at": consent.given_at.isoformat(),
                        "withdrawn_at": consent.withdrawn_at.isoformat() if consent.withdrawn_at else None
                    })
            
            logger.info(f"Access request processed for subject {subject.subject_id}")
            return {
                "status": "success",
                "request_type": "access",
                "data": subject_data,
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process access request: {str(e)}")
            return {"error": str(e)}
    
    def _process_rectification_request(self, subject: DataSubject, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Article 16 - Right to rectification"""
        try:
            updated_fields = []
            
            # Update personal data if provided
            if "name" in request_data and request_data["name"]:
                subject.name = request_data["name"]
                updated_fields.append("name")
            
            if "phone" in request_data and request_data["phone"]:
                subject.phone = request_data["phone"]
                updated_fields.append("phone")
            
            if "address" in request_data and request_data["address"]:
                subject.address = request_data["address"]
                updated_fields.append("address")
            
            logger.info(f"Rectification request processed for subject {subject.subject_id}: {updated_fields}")
            return {
                "status": "success",
                "request_type": "rectification",
                "updated_fields": updated_fields,
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process rectification request: {str(e)}")
            return {"error": str(e)}
    
    def _process_erasure_request(self, subject: DataSubject, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Article 17 - Right to erasure"""
        try:
            # Check if erasure is lawful
            can_erase = self._check_erasure_lawfulness(subject)
            
            if not can_erase:
                return {
                    "status": "rejected",
                    "request_type": "erasure",
                    "reason": "Erasure not lawful due to legal obligations or legitimate interests",
                    "processed_at": datetime.now().isoformat()
                }
            
            # Mark for erasure (in real implementation, this would trigger actual deletion)
            subject.consent_given = False
            subject.consent_date = None
            subject.data_categories = []
            subject.processing_purposes = []
            
            logger.info(f"Erasure request processed for subject {subject.subject_id}")
            return {
                "status": "success",
                "request_type": "erasure",
                "message": "Data marked for erasure",
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process erasure request: {str(e)}")
            return {"error": str(e)}
    
    def _process_portability_request(self, subject: DataSubject, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Article 20 - Right to data portability"""
        try:
            # Collect portable data
            portable_data = {
                "personal_data": {
                    "subject_id": subject.subject_id,
                    "email": subject.email,
                    "name": subject.name,
                    "phone": subject.phone,
                    "address": subject.address
                },
                "consent_records": [
                    {
                        "purpose": consent.purpose,
                        "data_categories": consent.data_categories,
                        "given_at": consent.given_at.isoformat()
                    }
                    for consent in self.consent_records.values()
                    if consent.subject_id == subject.subject_id and not consent.withdrawn_at
                ]
            }
            
            logger.info(f"Portability request processed for subject {subject.subject_id}")
            return {
                "status": "success",
                "request_type": "portability",
                "data": portable_data,
                "format": "json",
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process portability request: {str(e)}")
            return {"error": str(e)}
    
    def _process_objection_request(self, subject: DataSubject, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Article 21 - Right to object"""
        try:
            objection_reason = request_data.get("reason", "General objection")
            
            # In real implementation, this would stop processing for legitimate interests
            logger.info(f"Objection request processed for subject {subject.subject_id}: {objection_reason}")
            return {
                "status": "success",
                "request_type": "objection",
                "reason": objection_reason,
                "message": "Processing stopped for legitimate interests basis",
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process objection request: {str(e)}")
            return {"error": str(e)}
    
    def _check_erasure_lawfulness(self, subject: DataSubject) -> bool:
        """Check if data erasure is lawful"""
        try:
            # Check if subject has active processing activities with legal obligations
            for activity in self.processing_activities.values():
                if subject.subject_id in activity.data_subjects:
                    if activity.lawful_basis in ["legal_obligation", "vital_interests"]:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check erasure lawfulness: {str(e)}")
            return False
    
    def record_data_breach(self,
                          breach_type: str,
                          description: str,
                          affected_subjects: List[str],
                          severity: str = "medium") -> str:
        """Record a data breach incident"""
        try:
            breach_id = str(uuid.uuid4())
            
            breach_record = {
                "breach_id": breach_id,
                "breach_type": breach_type,
                "description": description,
                "affected_subjects": affected_subjects,
                "severity": severity,
                "discovered_at": datetime.now(),
                "reported_at": None,
                "status": "investigating"
            }
            
            self.breach_records.append(breach_record)
            
            # Check if notification is required (within 72 hours)
            if severity in ["high", "critical"]:
                self._trigger_breach_notification(breach_record)
            
            logger.info(f"Data breach recorded: {breach_id}")
            return breach_id
            
        except Exception as e:
            logger.error(f"Failed to record data breach: {str(e)}")
            return ""
    
    def _trigger_breach_notification(self, breach_record: Dict[str, Any]) -> None:
        """Trigger breach notification to supervisory authority"""
        try:
            # In real implementation, this would send actual notifications
            breach_record["reported_at"] = datetime.now()
            breach_record["status"] = "reported"
            
            logger.warning(f"Breach notification triggered for breach {breach_record['breach_id']}")
            
        except Exception as e:
            logger.error(f"Failed to trigger breach notification: {str(e)}")
    
    def generate_privacy_impact_assessment(self, activity_id: str) -> Dict[str, Any]:
        """Generate a Data Protection Impact Assessment (DPIA)"""
        try:
            if activity_id not in self.processing_activities:
                raise ValueError(f"Processing activity {activity_id} not found")
            
            activity = self.processing_activities[activity_id]
            
            dpia = {
                "activity_id": activity_id,
                "activity_name": activity.name,
                "purpose": activity.purpose,
                "lawful_basis": activity.lawful_basis,
                "data_categories": activity.data_categories,
                "risk_assessment": {
                    "privacy_risks": self._assess_privacy_risks(activity),
                    "mitigation_measures": self._recommend_mitigations(activity),
                    "risk_level": self._calculate_risk_level(activity)
                },
                "compliance_status": self._check_compliance_status(activity),
                "recommendations": self._generate_recommendations(activity),
                "generated_at": datetime.now().isoformat()
            }
            
            logger.info(f"DPIA generated for activity {activity_id}")
            return dpia
            
        except Exception as e:
            logger.error(f"Failed to generate DPIA: {str(e)}")
            return {"error": str(e)}
    
    def _assess_privacy_risks(self, activity: ProcessingActivity) -> List[Dict[str, Any]]:
        """Assess privacy risks for a processing activity"""
        risks = []
        
        # High-risk data categories
        high_risk_categories = ["health", "biometric", "genetic", "political", "religious"]
        for category in activity.data_categories:
            if any(risk_cat in category.lower() for risk_cat in high_risk_categories):
                risks.append({
                    "risk": "High-risk data processing",
                    "description": f"Processing {category} data",
                    "severity": "high"
                })
        
        # Large-scale processing
        if len(activity.data_subjects) > 1000:
            risks.append({
                "risk": "Large-scale processing",
                "description": f"Processing data of {len(activity.data_subjects)} subjects",
                "severity": "medium"
            })
        
        return risks
    
    def _recommend_mitigations(self, activity: ProcessingActivity) -> List[str]:
        """Recommend mitigation measures"""
        mitigations = [
            "Implement data minimization principles",
            "Use pseudonymization where possible",
            "Implement access controls and encryption",
            "Regular security assessments",
            "Data subject rights automation"
        ]
        
        if len(activity.data_subjects) > 1000:
            mitigations.append("Appoint a Data Protection Officer")
        
        return mitigations
    
    def _calculate_risk_level(self, activity: ProcessingActivity) -> str:
        """Calculate overall risk level"""
        risk_score = 0
        
        # High-risk data categories
        high_risk_categories = ["health", "biometric", "genetic", "political", "religious"]
        for category in activity.data_categories:
            if any(risk_cat in category.lower() for risk_cat in high_risk_categories):
                risk_score += 3
        
        # Large-scale processing
        if len(activity.data_subjects) > 1000:
            risk_score += 2
        
        # Automated decision making
        if "automated" in activity.purpose.lower():
            risk_score += 2
        
        if risk_score >= 5:
            return "high"
        elif risk_score >= 3:
            return "medium"
        else:
            return "low"
    
    def _check_compliance_status(self, activity: ProcessingActivity) -> Dict[str, bool]:
        """Check compliance status for various GDPR requirements"""
        return {
            "lawful_basis_documented": bool(activity.lawful_basis),
            "purpose_specified": bool(activity.purpose),
            "retention_period_set": activity.retention_period > 0,
            "data_categories_defined": len(activity.data_categories) > 0,
            "consent_mechanism": any(consent.subject_id in activity.data_subjects 
                                   for consent in self.consent_records.values())
        }
    
    def _generate_recommendations(self, activity: ProcessingActivity) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        if not activity.lawful_basis:
            recommendations.append("Document lawful basis for processing")
        
        if activity.retention_period <= 0:
            recommendations.append("Set appropriate retention period")
        
        if len(activity.data_categories) == 0:
            recommendations.append("Define data categories being processed")
        
        return recommendations
    
    def _generate_subject_id(self, email: str) -> str:
        """Generate a unique subject ID"""
        return hashlib.sha256(email.encode()).hexdigest()[:16]
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive compliance report"""
        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "data_subjects": {
                    "total": len(self.data_subjects),
                    "with_consent": len([s for s in self.data_subjects.values() if s.consent_given]),
                    "without_consent": len([s for s in self.data_subjects.values() if not s.consent_given])
                },
                "processing_activities": {
                    "total": len(self.processing_activities),
                    "by_lawful_basis": {}
                },
                "consent_records": {
                    "total": len(self.consent_records),
                    "active": len([c for c in self.consent_records.values() if not c.withdrawn_at]),
                    "withdrawn": len([c for c in self.consent_records.values() if c.withdrawn_at])
                },
                "data_breaches": {
                    "total": len(self.breach_records),
                    "by_severity": {},
                    "reported": len([b for b in self.breach_records if b.get("reported_at")])
                }
            }
            
            # Count by lawful basis
            for activity in self.processing_activities.values():
                basis = activity.lawful_basis
                report["processing_activities"]["by_lawful_basis"][basis] = \
                    report["processing_activities"]["by_lawful_basis"].get(basis, 0) + 1
            
            # Count breaches by severity
            for breach in self.breach_records:
                severity = breach["severity"]
                report["data_breaches"]["by_severity"][severity] = \
                    report["data_breaches"]["by_severity"].get(severity, 0) + 1
            
            logger.info("Compliance report generated")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {str(e)}")
            return {"error": str(e)}
