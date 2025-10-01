"""
Privacy Controller
Manages privacy controls and PII detection/management
"""

import logging
import re
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class PIIType(Enum):
    """Types of Personally Identifiable Information"""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    DRIVER_LICENSE = "driver_license"
    PASSPORT = "passport"

@dataclass
class PIIMatch:
    """Represents a PII match"""
    pii_type: PIIType
    value: str
    masked_value: str
    position: Tuple[int, int]
    confidence: float
    context: str

@dataclass
class PrivacyPolicy:
    """Represents a privacy policy"""
    policy_id: str
    name: str
    version: str
    effective_date: datetime
    data_categories: List[str]
    processing_purposes: List[str]
    retention_period: int
    sharing_policy: str
    contact_info: str

class PrivacyController:
    """
    Manages privacy controls and PII detection/management
    """
    
    def __init__(self):
        self.pii_patterns = self._initialize_pii_patterns()
        self.privacy_policies = {}
        self.consent_preferences = {}
        self.data_classifications = {}
        
        logger.info("PrivacyController initialized")
    
    def _initialize_pii_patterns(self) -> Dict[PIIType, str]:
        """Initialize regex patterns for PII detection"""
        return {
            PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            PIIType.PHONE: r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            PIIType.SSN: r'\b\d{3}-?\d{2}-?\d{4}\b',
            PIIType.CREDIT_CARD: r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            PIIType.IP_ADDRESS: r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            PIIType.MAC_ADDRESS: r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b',
            PIIType.DATE_OF_BIRTH: r'\b(0?[1-9]|1[0-2])[-/](0?[1-9]|[12][0-9]|3[01])[-/](19|20)\d{2}\b',
            PIIType.DRIVER_LICENSE: r'\b[A-Z]\d{7,8}\b',
            PIIType.PASSPORT: r'\b[A-Z]{1,2}\d{6,9}\b'
        }
    
    def detect_pii(self, text: str, context: str = "") -> List[PIIMatch]:
        """Detect PII in text"""
        try:
            matches = []
            
            for pii_type, pattern in self.pii_patterns.items():
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    value = match.group()
                    masked_value = self._mask_pii(value, pii_type)
                    
                    pii_match = PIIMatch(
                        pii_type=pii_type,
                        value=value,
                        masked_value=masked_value,
                        position=(match.start(), match.end()),
                        confidence=self._calculate_confidence(value, pii_type),
                        context=context
                    )
                    
                    matches.append(pii_match)
            
            logger.info(f"Detected {len(matches)} PII matches in text")
            return matches
            
        except Exception as e:
            logger.error(f"Failed to detect PII: {str(e)}")
            return []
    
    def _mask_pii(self, value: str, pii_type: PIIType) -> str:
        """Mask PII value based on type"""
        try:
            if pii_type == PIIType.EMAIL:
                # Mask email: user@domain.com -> u***@d***.com
                parts = value.split('@')
                if len(parts) == 2:
                    username = parts[0]
                    domain = parts[1]
                    masked_username = username[0] + '*' * (len(username) - 1)
                    masked_domain = domain[0] + '*' * (len(domain) - 1)
                    return f"{masked_username}@{masked_domain}"
            
            elif pii_type == PIIType.PHONE:
                # Mask phone: (555) 123-4567 -> (***) ***-****
                return re.sub(r'\d', '*', value)
            
            elif pii_type == PIIType.SSN:
                # Mask SSN: 123-45-6789 -> ***-**-****
                return re.sub(r'\d', '*', value)
            
            elif pii_type == PIIType.CREDIT_CARD:
                # Mask credit card: 1234 5678 9012 3456 -> **** **** **** 3456
                digits = re.findall(r'\d', value)
                if len(digits) >= 4:
                    return '**** ' * (len(digits) // 4 - 1) + ''.join(digits[-4:])
            
            elif pii_type == PIIType.IP_ADDRESS:
                # Mask IP: 192.168.1.1 -> 192.168.*.*
                parts = value.split('.')
                if len(parts) == 4:
                    return '.'.join(parts[:2] + ['*', '*'])
            
            else:
                # Default masking: replace with asterisks
                return '*' * len(value)
            
            return value
            
        except Exception as e:
            logger.error(f"Failed to mask PII: {str(e)}")
            return value
    
    def _calculate_confidence(self, value: str, pii_type: PIIType) -> float:
        """Calculate confidence score for PII detection"""
        try:
            base_confidence = 0.8
            
            # Adjust confidence based on value characteristics
            if pii_type == PIIType.EMAIL:
                if '@' in value and '.' in value.split('@')[1]:
                    base_confidence = 0.95
                else:
                    base_confidence = 0.7
            
            elif pii_type == PIIType.PHONE:
                digits = re.findall(r'\d', value)
                if len(digits) == 10:
                    base_confidence = 0.9
                elif len(digits) == 11 and digits[0] == '1':
                    base_confidence = 0.9
                else:
                    base_confidence = 0.6
            
            elif pii_type == PIIType.SSN:
                digits = re.findall(r'\d', value)
                if len(digits) == 9:
                    base_confidence = 0.9
                else:
                    base_confidence = 0.5
            
            return min(base_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence: {str(e)}")
            return 0.5
    
    def anonymize_data(self, data: Dict[str, Any], anonymization_level: str = "medium") -> Dict[str, Any]:
        """Anonymize data based on specified level"""
        try:
            anonymized_data = {}
            
            for key, value in data.items():
                if isinstance(value, str):
                    # Detect and mask PII in string values
                    pii_matches = self.detect_pii(value, key)
                    if pii_matches:
                        anonymized_value = value
                        for match in reversed(pii_matches):  # Reverse to maintain positions
                            start, end = match.position
                            anonymized_value = anonymized_value[:start] + match.masked_value + anonymized_value[end:]
                        anonymized_data[key] = anonymized_value
                    else:
                        anonymized_data[key] = value
                elif isinstance(value, dict):
                    anonymized_data[key] = self.anonymize_data(value, anonymization_level)
                elif isinstance(value, list):
                    anonymized_data[key] = [
                        self.anonymize_data(item, anonymization_level) if isinstance(item, dict)
                        else self._anonymize_primitive(item, anonymization_level)
                        for item in value
                    ]
                else:
                    anonymized_data[key] = self._anonymize_primitive(value, anonymization_level)
            
            logger.info(f"Data anonymized with level: {anonymization_level}")
            return anonymized_data
            
        except Exception as e:
            logger.error(f"Failed to anonymize data: {str(e)}")
            return data
    
    def _anonymize_primitive(self, value: Any, anonymization_level: str) -> Any:
        """Anonymize primitive values"""
        try:
            if anonymization_level == "high":
                # High anonymization: replace with hash
                if isinstance(value, (str, int, float)):
                    return hashlib.sha256(str(value).encode()).hexdigest()[:8]
            elif anonymization_level == "medium":
                # Medium anonymization: mask partially
                if isinstance(value, str) and len(value) > 3:
                    return value[0] + '*' * (len(value) - 2) + value[-1]
            # Low anonymization: keep as is
            return value
            
        except Exception as e:
            logger.error(f"Failed to anonymize primitive: {str(e)}")
            return value
    
    def pseudonymize_data(self, data: Dict[str, Any], salt: str = "") -> Dict[str, Any]:
        """Pseudonymize data using consistent hashing"""
        try:
            pseudonymized_data = {}
            
            for key, value in data.items():
                if isinstance(value, str) and self._is_pii_field(key):
                    # Use consistent hashing for pseudonymization
                    hash_input = f"{value}{salt}".encode()
                    pseudonymized_value = hashlib.sha256(hash_input).hexdigest()[:16]
                    pseudonymized_data[key] = pseudonymized_value
                elif isinstance(value, dict):
                    pseudonymized_data[key] = self.pseudonymize_data(value, salt)
                elif isinstance(value, list):
                    pseudonymized_data[key] = [
                        self.pseudonymize_data(item, salt) if isinstance(item, dict)
                        else self._pseudonymize_primitive(item, salt, key)
                        for item in value
                    ]
                else:
                    pseudonymized_data[key] = self._pseudonymize_primitive(value, salt, key)
            
            logger.info("Data pseudonymized")
            return pseudonymized_data
            
        except Exception as e:
            logger.error(f"Failed to pseudonymize data: {str(e)}")
            return data
    
    def _is_pii_field(self, field_name: str) -> bool:
        """Check if field name indicates PII"""
        pii_indicators = [
            'email', 'phone', 'ssn', 'name', 'address', 'id', 'user_id',
            'customer_id', 'personal', 'private', 'sensitive'
        ]
        
        field_lower = field_name.lower()
        return any(indicator in field_lower for indicator in pii_indicators)
    
    def _pseudonymize_primitive(self, value: Any, salt: str, field_name: str) -> Any:
        """Pseudonymize primitive values"""
        try:
            if isinstance(value, (str, int, float)) and self._is_pii_field(field_name):
                hash_input = f"{value}{salt}".encode()
                return hashlib.sha256(hash_input).hexdigest()[:16]
            return value
            
        except Exception as e:
            logger.error(f"Failed to pseudonymize primitive: {str(e)}")
            return value
    
    def create_privacy_policy(self,
                            name: str,
                            data_categories: List[str],
                            processing_purposes: List[str],
                            retention_period: int,
                            sharing_policy: str,
                            contact_info: str) -> str:
        """Create a privacy policy"""
        try:
            policy_id = f"policy_{int(datetime.now().timestamp())}"
            
            policy = PrivacyPolicy(
                policy_id=policy_id,
                name=name,
                version="1.0",
                effective_date=datetime.now(),
                data_categories=data_categories,
                processing_purposes=processing_purposes,
                retention_period=retention_period,
                sharing_policy=sharing_policy,
                contact_info=contact_info
            )
            
            self.privacy_policies[policy_id] = policy
            
            logger.info(f"Privacy policy created: {policy_id}")
            return policy_id
            
        except Exception as e:
            logger.error(f"Failed to create privacy policy: {str(e)}")
            return ""
    
    def set_consent_preferences(self,
                              subject_id: str,
                              preferences: Dict[str, Any]) -> bool:
        """Set consent preferences for a data subject"""
        try:
            self.consent_preferences[subject_id] = {
                "preferences": preferences,
                "updated_at": datetime.now(),
                "version": "1.0"
            }
            
            logger.info(f"Consent preferences set for subject {subject_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set consent preferences: {str(e)}")
            return False
    
    def get_consent_preferences(self, subject_id: str) -> Optional[Dict[str, Any]]:
        """Get consent preferences for a data subject"""
        try:
            return self.consent_preferences.get(subject_id)
            
        except Exception as e:
            logger.error(f"Failed to get consent preferences: {str(e)}")
            return None
    
    def classify_data_sensitivity(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Classify data sensitivity levels"""
        try:
            classifications = {}
            
            for key, value in data.items():
                if isinstance(value, str):
                    pii_matches = self.detect_pii(value, key)
                    if pii_matches:
                        # Classify based on PII types found
                        high_sensitivity_types = [PIIType.SSN, PIIType.CREDIT_CARD, PIIType.DRIVER_LICENSE]
                        if any(match.pii_type in high_sensitivity_types for match in pii_matches):
                            classifications[key] = "high"
                        else:
                            classifications[key] = "medium"
                    else:
                        classifications[key] = "low"
                elif isinstance(value, dict):
                    classifications[key] = self.classify_data_sensitivity(value)
                else:
                    classifications[key] = "low"
            
            return classifications
            
        except Exception as e:
            logger.error(f"Failed to classify data sensitivity: {str(e)}")
            return {}
    
    def generate_privacy_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a privacy analysis report"""
        try:
            # Detect all PII
            pii_matches = []
            for key, value in data.items():
                if isinstance(value, str):
                    matches = self.detect_pii(value, key)
                    pii_matches.extend(matches)
            
            # Classify sensitivity
            sensitivity_classifications = self.classify_data_sensitivity(data)
            
            # Generate report
            report = {
                "generated_at": datetime.now().isoformat(),
                "pii_detection": {
                    "total_matches": len(pii_matches),
                    "by_type": {},
                    "high_confidence": len([m for m in pii_matches if m.confidence > 0.8]),
                    "medium_confidence": len([m for m in pii_matches if 0.5 < m.confidence <= 0.8]),
                    "low_confidence": len([m for m in pii_matches if m.confidence <= 0.5])
                },
                "sensitivity_analysis": {
                    "high_sensitivity_fields": [k for k, v in sensitivity_classifications.items() if v == "high"],
                    "medium_sensitivity_fields": [k for k, v in sensitivity_classifications.items() if v == "medium"],
                    "low_sensitivity_fields": [k for k, v in sensitivity_classifications.items() if v == "low"]
                },
                "recommendations": self._generate_privacy_recommendations(pii_matches, sensitivity_classifications)
            }
            
            # Count PII by type
            for match in pii_matches:
                pii_type = match.pii_type.value
                report["pii_detection"]["by_type"][pii_type] = \
                    report["pii_detection"]["by_type"].get(pii_type, 0) + 1
            
            logger.info("Privacy report generated")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate privacy report: {str(e)}")
            return {"error": str(e)}
    
    def _generate_privacy_recommendations(self, 
                                        pii_matches: List[PIIMatch],
                                        classifications: Dict[str, str]) -> List[str]:
        """Generate privacy recommendations"""
        recommendations = []
        
        if pii_matches:
            recommendations.append("PII detected - consider data minimization")
            recommendations.append("Implement proper consent mechanisms")
            recommendations.append("Ensure data subject rights are supported")
        
        high_sensitivity_count = len([v for v in classifications.values() if v == "high"])
        if high_sensitivity_count > 0:
            recommendations.append(f"{high_sensitivity_count} high-sensitivity fields - implement strong security controls")
            recommendations.append("Consider data protection impact assessment")
        
        if any(match.pii_type == PIIType.CREDIT_CARD for match in pii_matches):
            recommendations.append("Credit card data detected - ensure PCI DSS compliance")
        
        if any(match.pii_type == PIIType.SSN for match in pii_matches):
            recommendations.append("SSN detected - implement strict access controls")
        
        return recommendations
    
    def validate_privacy_compliance(self, 
                                  data: Dict[str, Any],
                                  policy_id: str) -> Dict[str, Any]:
        """Validate data against privacy policy"""
        try:
            if policy_id not in self.privacy_policies:
                raise ValueError(f"Privacy policy {policy_id} not found")
            
            policy = self.privacy_policies[policy_id]
            
            # Check data categories
            detected_categories = self._detect_data_categories(data)
            policy_categories = set(policy.data_categories)
            category_compliance = detected_categories.issubset(policy_categories)
            
            # Check for unexpected PII
            pii_matches = []
            for key, value in data.items():
                if isinstance(value, str):
                    matches = self.detect_pii(value, key)
                    pii_matches.extend(matches)
            
            unexpected_pii = [m for m in pii_matches if m.pii_type.value not in policy.data_categories]
            
            compliance_result = {
                "policy_id": policy_id,
                "compliant": category_compliance and len(unexpected_pii) == 0,
                "data_categories": {
                    "detected": list(detected_categories),
                    "policy_allowed": policy.data_categories,
                    "compliant": category_compliance
                },
                "pii_detection": {
                    "total_matches": len(pii_matches),
                    "unexpected_pii": len(unexpected_pii),
                    "unexpected_types": [m.pii_type.value for m in unexpected_pii]
                },
                "recommendations": []
            }
            
            if not category_compliance:
                compliance_result["recommendations"].append("Data categories not covered by policy")
            
            if unexpected_pii:
                compliance_result["recommendations"].append("Unexpected PII types detected")
            
            logger.info(f"Privacy compliance validation completed for policy {policy_id}")
            return compliance_result
            
        except Exception as e:
            logger.error(f"Failed to validate privacy compliance: {str(e)}")
            return {"error": str(e)}
    
    def _detect_data_categories(self, data: Dict[str, Any]) -> set:
        """Detect data categories in the data"""
        categories = set()
        
        for key, value in data.items():
            if isinstance(value, str):
                pii_matches = self.detect_pii(value, key)
                for match in pii_matches:
                    categories.add(match.pii_type.value)
            elif isinstance(value, dict):
                categories.update(self._detect_data_categories(value))
        
        return categories
