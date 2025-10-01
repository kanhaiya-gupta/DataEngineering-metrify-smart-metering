"""
Consent Manager
Manages consent collection, tracking, and withdrawal
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid
import json

logger = logging.getLogger(__name__)

class ConsentStatus(Enum):
    """Consent status values"""
    GIVEN = "given"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    PENDING = "pending"

class ConsentType(Enum):
    """Types of consent"""
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    OPT_IN = "opt_in"
    OPT_OUT = "opt_out"

@dataclass
class ConsentRecord:
    """Represents a consent record"""
    consent_id: str
    subject_id: str
    consent_type: ConsentType
    purpose: str
    data_categories: List[str]
    status: ConsentStatus
    given_at: datetime
    withdrawn_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    consent_text: Optional[str] = None
    version: str = "1.0"

class ConsentManager:
    """
    Manages consent collection, tracking, and withdrawal
    """
    
    def __init__(self):
        self.consent_records = {}
        self.consent_templates = {}
        self.consent_preferences = {}
        
        logger.info("ConsentManager initialized")
    
    def create_consent_template(self,
                              template_name: str,
                              purpose: str,
                              data_categories: List[str],
                              consent_text: str,
                              consent_type: ConsentType = ConsentType.EXPLICIT,
                              expires_days: Optional[int] = None) -> str:
        """Create a consent template"""
        try:
            template_id = str(uuid.uuid4())
            
            template = {
                "template_id": template_id,
                "template_name": template_name,
                "purpose": purpose,
                "data_categories": data_categories,
                "consent_text": consent_text,
                "consent_type": consent_type,
                "expires_days": expires_days,
                "created_at": datetime.now(),
                "version": "1.0"
            }
            
            self.consent_templates[template_id] = template
            
            logger.info(f"Consent template created: {template_id}")
            return template_id
            
        except Exception as e:
            logger.error(f"Failed to create consent template: {str(e)}")
            return ""
    
    def collect_consent(self,
                       subject_id: str,
                       template_id: str,
                       ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None,
                       custom_purpose: Optional[str] = None) -> str:
        """Collect consent from a data subject"""
        try:
            if template_id not in self.consent_templates:
                raise ValueError(f"Consent template {template_id} not found")
            
            template = self.consent_templates[template_id]
            
            consent_id = str(uuid.uuid4())
            
            # Calculate expiration date
            expires_at = None
            if template["expires_days"]:
                expires_at = datetime.now() + timedelta(days=template["expires_days"])
            
            consent = ConsentRecord(
                consent_id=consent_id,
                subject_id=subject_id,
                consent_type=template["consent_type"],
                purpose=custom_purpose or template["purpose"],
                data_categories=template["data_categories"],
                status=ConsentStatus.GIVEN,
                given_at=datetime.now(),
                expires_at=expires_at,
                ip_address=ip_address,
                user_agent=user_agent,
                consent_text=template["consent_text"],
                version=template["version"]
            )
            
            self.consent_records[consent_id] = consent
            
            logger.info(f"Consent collected: {consent_id} for subject {subject_id}")
            return consent_id
            
        except Exception as e:
            logger.error(f"Failed to collect consent: {str(e)}")
            return ""
    
    def withdraw_consent(self, consent_id: str) -> bool:
        """Withdraw consent"""
        try:
            if consent_id not in self.consent_records:
                raise ValueError(f"Consent {consent_id} not found")
            
            consent = self.consent_records[consent_id]
            consent.status = ConsentStatus.WITHDRAWN
            consent.withdrawn_at = datetime.now()
            
            logger.info(f"Consent withdrawn: {consent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to withdraw consent: {str(e)}")
            return False
    
    def check_consent_validity(self, subject_id: str, purpose: str, data_categories: List[str]) -> bool:
        """Check if valid consent exists for specific purpose and data categories"""
        try:
            valid_consents = []
            
            for consent in self.consent_records.values():
                if (consent.subject_id == subject_id and
                    consent.status == ConsentStatus.GIVEN and
                    consent.purpose == purpose and
                    all(cat in consent.data_categories for cat in data_categories)):
                    
                    # Check if consent has expired
                    if consent.expires_at and consent.expires_at < datetime.now():
                        consent.status = ConsentStatus.EXPIRED
                        continue
                    
                    valid_consents.append(consent)
            
            has_valid_consent = len(valid_consents) > 0
            logger.debug(f"Consent validity check for subject {subject_id}: {has_valid_consent}")
            return has_valid_consent
            
        except Exception as e:
            logger.error(f"Failed to check consent validity: {str(e)}")
            return False
    
    def get_subject_consents(self, subject_id: str) -> List[ConsentRecord]:
        """Get all consents for a data subject"""
        try:
            subject_consents = [
                consent for consent in self.consent_records.values()
                if consent.subject_id == subject_id
            ]
            
            logger.debug(f"Retrieved {len(subject_consents)} consents for subject {subject_id}")
            return subject_consents
            
        except Exception as e:
            logger.error(f"Failed to get subject consents: {str(e)}")
            return []
    
    def update_consent_preferences(self,
                                 subject_id: str,
                                 preferences: Dict[str, Any]) -> bool:
        """Update consent preferences for a data subject"""
        try:
            self.consent_preferences[subject_id] = {
                "preferences": preferences,
                "updated_at": datetime.now(),
                "version": "1.0"
            }
            
            logger.info(f"Consent preferences updated for subject {subject_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update consent preferences: {str(e)}")
            return False
    
    def get_consent_statistics(self) -> Dict[str, Any]:
        """Get consent statistics"""
        try:
            total_consents = len(self.consent_records)
            
            status_counts = {}
            type_counts = {}
            purpose_counts = {}
            
            for consent in self.consent_records.values():
                # Count by status
                status = consent.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Count by type
                consent_type = consent.consent_type.value
                type_counts[consent_type] = type_counts.get(consent_type, 0) + 1
                
                # Count by purpose
                purpose = consent.purpose
                purpose_counts[purpose] = purpose_counts.get(purpose, 0) + 1
            
            return {
                "total_consents": total_consents,
                "status_counts": status_counts,
                "type_counts": type_counts,
                "purpose_counts": purpose_counts,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get consent statistics: {str(e)}")
            return {"error": str(e)}
