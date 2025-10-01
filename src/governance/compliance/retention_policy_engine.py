"""
Retention Policy Engine
Manages data retention policies and automated data lifecycle
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid
import json

logger = logging.getLogger(__name__)

class RetentionAction(Enum):
    """Retention actions"""
    DELETE = "delete"
    ARCHIVE = "archive"
    ANONYMIZE = "anonymize"
    REVIEW = "review"
    EXTEND = "extend"

class DataCategory(Enum):
    """Data categories for retention policies"""
    PERSONAL_DATA = "personal_data"
    BUSINESS_DATA = "business_data"
    AUDIT_LOGS = "audit_logs"
    SYSTEM_LOGS = "system_logs"
    BACKUP_DATA = "backup_data"
    ANALYTICS_DATA = "analytics_data"
    ML_MODELS = "ml_models"

@dataclass
class RetentionPolicy:
    """Represents a data retention policy"""
    policy_id: str
    name: str
    data_categories: List[DataCategory]
    retention_period_days: int
    action: RetentionAction
    conditions: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    is_active: bool = True

@dataclass
class DataItem:
    """Represents a data item subject to retention"""
    item_id: str
    data_category: DataCategory
    created_at: datetime
    last_accessed: datetime
    size_bytes: int
    location: str
    metadata: Dict[str, Any]
    retention_policy_id: Optional[str] = None
    scheduled_action: Optional[RetentionAction] = None
    scheduled_date: Optional[datetime] = None

class RetentionPolicyEngine:
    """
    Manages data retention policies and automated data lifecycle
    """
    
    def __init__(self):
        self.retention_policies = {}
        self.data_items = {}
        self.retention_actions = []
        
        logger.info("RetentionPolicyEngine initialized")
    
    def create_retention_policy(self,
                              name: str,
                              data_categories: List[DataCategory],
                              retention_period_days: int,
                              action: RetentionAction,
                              conditions: Dict[str, Any] = None) -> str:
        """Create a new retention policy"""
        try:
            policy_id = str(uuid.uuid4())
            
            policy = RetentionPolicy(
                policy_id=policy_id,
                name=name,
                data_categories=data_categories,
                retention_period_days=retention_period_days,
                action=action,
                conditions=conditions or {},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.retention_policies[policy_id] = policy
            
            logger.info(f"Retention policy created: {policy_id}")
            return policy_id
            
        except Exception as e:
            logger.error(f"Failed to create retention policy: {str(e)}")
            return ""
    
    def register_data_item(self,
                          data_category: DataCategory,
                          created_at: datetime,
                          size_bytes: int,
                          location: str,
                          metadata: Dict[str, Any] = None) -> str:
        """Register a data item for retention tracking"""
        try:
            item_id = str(uuid.uuid4())
            
            data_item = DataItem(
                item_id=item_id,
                data_category=data_category,
                created_at=created_at,
                last_accessed=created_at,
                size_bytes=size_bytes,
                location=location,
                metadata=metadata or {}
            )
            
            # Find applicable retention policy
            applicable_policy = self._find_applicable_policy(data_item)
            if applicable_policy:
                data_item.retention_policy_id = applicable_policy.policy_id
                data_item.scheduled_action = applicable_policy.action
                data_item.scheduled_date = created_at + timedelta(days=applicable_policy.retention_period_days)
            
            self.data_items[item_id] = data_item
            
            logger.info(f"Data item registered: {item_id}")
            return item_id
            
        except Exception as e:
            logger.error(f"Failed to register data item: {str(e)}")
            return ""
    
    def _find_applicable_policy(self, data_item: DataItem) -> Optional[RetentionPolicy]:
        """Find applicable retention policy for a data item"""
        try:
            applicable_policies = []
            
            for policy in self.retention_policies.values():
                if not policy.is_active:
                    continue
                
                # Check if data category matches
                if data_item.data_category in policy.data_categories:
                    # Check additional conditions
                    if self._check_policy_conditions(data_item, policy):
                        applicable_policies.append(policy)
            
            # Return the most specific policy (most conditions matched)
            if applicable_policies:
                return max(applicable_policies, key=lambda p: len(p.conditions))
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find applicable policy: {str(e)}")
            return None
    
    def _check_policy_conditions(self, data_item: DataItem, policy: RetentionPolicy) -> bool:
        """Check if data item meets policy conditions"""
        try:
            conditions = policy.conditions
            
            # Check size conditions
            if "min_size_bytes" in conditions:
                if data_item.size_bytes < conditions["min_size_bytes"]:
                    return False
            
            if "max_size_bytes" in conditions:
                if data_item.size_bytes > conditions["max_size_bytes"]:
                    return False
            
            # Check metadata conditions
            if "required_metadata" in conditions:
                required_keys = conditions["required_metadata"]
                if not all(key in data_item.metadata for key in required_keys):
                    return False
            
            # Check location patterns
            if "location_patterns" in conditions:
                location_patterns = conditions["location_patterns"]
                if not any(pattern in data_item.location for pattern in location_patterns):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check policy conditions: {str(e)}")
            return False
    
    def process_retention_actions(self) -> Dict[str, Any]:
        """Process all due retention actions"""
        try:
            now = datetime.now()
            processed_actions = []
            
            for item_id, data_item in self.data_items.items():
                if (data_item.scheduled_action and 
                    data_item.scheduled_date and 
                    data_item.scheduled_date <= now):
                    
                    action_result = self._execute_retention_action(data_item)
                    processed_actions.append({
                        "item_id": item_id,
                        "action": data_item.scheduled_action.value,
                        "result": action_result,
                        "processed_at": now.isoformat()
                    })
                    
                    # Record the action
                    self.retention_actions.append({
                        "action_id": str(uuid.uuid4()),
                        "item_id": item_id,
                        "action": data_item.scheduled_action.value,
                        "processed_at": now,
                        "result": action_result
                    })
            
            logger.info(f"Processed {len(processed_actions)} retention actions")
            return {
                "processed_actions": processed_actions,
                "total_processed": len(processed_actions),
                "processed_at": now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process retention actions: {str(e)}")
            return {"error": str(e)}
    
    def _execute_retention_action(self, data_item: DataItem) -> str:
        """Execute retention action on data item"""
        try:
            action = data_item.scheduled_action
            
            if action == RetentionAction.DELETE:
                # In real implementation, this would delete the actual data
                logger.info(f"Data item {data_item.item_id} marked for deletion")
                return "deleted"
            
            elif action == RetentionAction.ARCHIVE:
                # In real implementation, this would move data to archive
                logger.info(f"Data item {data_item.item_id} archived")
                return "archived"
            
            elif action == RetentionAction.ANONYMIZE:
                # In real implementation, this would anonymize the data
                logger.info(f"Data item {data_item.item_id} anonymized")
                return "anonymized"
            
            elif action == RetentionAction.REVIEW:
                # In real implementation, this would flag for manual review
                logger.info(f"Data item {data_item.item_id} flagged for review")
                return "flagged_for_review"
            
            elif action == RetentionAction.EXTEND:
                # Extend retention period
                if data_item.retention_policy_id in self.retention_policies:
                    policy = self.retention_policies[data_item.retention_policy_id]
                    data_item.scheduled_date = datetime.now() + timedelta(days=policy.retention_period_days)
                    logger.info(f"Retention extended for data item {data_item.item_id}")
                    return "extended"
            
            return "no_action"
            
        except Exception as e:
            logger.error(f"Failed to execute retention action: {str(e)}")
            return "error"
    
    def get_items_due_for_action(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """Get data items due for retention action in the next N days"""
        try:
            cutoff_date = datetime.now() + timedelta(days=days_ahead)
            
            due_items = []
            for item_id, data_item in self.data_items.items():
                if (data_item.scheduled_action and 
                    data_item.scheduled_date and 
                    data_item.scheduled_date <= cutoff_date):
                    
                    due_items.append({
                        "item_id": item_id,
                        "data_category": data_item.data_category.value,
                        "created_at": data_item.created_at.isoformat(),
                        "scheduled_action": data_item.scheduled_action.value,
                        "scheduled_date": data_item.scheduled_date.isoformat(),
                        "location": data_item.location,
                        "size_bytes": data_item.size_bytes
                    })
            
            # Sort by scheduled date
            due_items.sort(key=lambda x: x["scheduled_date"])
            
            logger.info(f"Found {len(due_items)} items due for action in next {days_ahead} days")
            return due_items
            
        except Exception as e:
            logger.error(f"Failed to get items due for action: {str(e)}")
            return []
    
    def update_data_item_access(self, item_id: str) -> bool:
        """Update last accessed time for a data item"""
        try:
            if item_id not in self.data_items:
                raise ValueError(f"Data item {item_id} not found")
            
            data_item = self.data_items[item_id]
            data_item.last_accessed = datetime.now()
            
            logger.debug(f"Updated access time for data item {item_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update data item access: {str(e)}")
            return False
    
    def extend_retention(self, item_id: str, additional_days: int) -> bool:
        """Extend retention period for a data item"""
        try:
            if item_id not in self.data_items:
                raise ValueError(f"Data item {item_id} not found")
            
            data_item = self.data_items[item_id]
            if data_item.scheduled_date:
                data_item.scheduled_date += timedelta(days=additional_days)
                logger.info(f"Extended retention for item {item_id} by {additional_days} days")
                return True
            else:
                logger.warning(f"No scheduled action for item {item_id}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to extend retention: {str(e)}")
            return False
    
    def get_retention_report(self) -> Dict[str, Any]:
        """Generate retention policy report"""
        try:
            total_items = len(self.data_items)
            total_policies = len(self.retention_policies)
            
            # Count items by category
            category_counts = {}
            for data_item in self.data_items.values():
                category = data_item.data_category.value
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Count items by action
            action_counts = {}
            for data_item in self.data_items.values():
                if data_item.scheduled_action:
                    action = data_item.scheduled_action.value
                    action_counts[action] = action_counts.get(action, 0) + 1
            
            # Count items due soon
            due_soon = len(self.get_items_due_for_action(days_ahead=30))
            
            # Calculate total storage
            total_storage = sum(item.size_bytes for item in self.data_items.values())
            
            report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_items": total_items,
                    "total_policies": total_policies
                },
                "item_statistics": {
                    "category_counts": category_counts,
                    "action_counts": action_counts,
                    "due_soon_30_days": due_soon,
                    "total_storage_bytes": total_storage,
                    "total_storage_gb": round(total_storage / (1024**3), 2)
                },
                "policy_statistics": {
                    "active_policies": len([p for p in self.retention_policies.values() if p.is_active]),
                    "inactive_policies": len([p for p in self.retention_policies.values() if not p.is_active])
                }
            }
            
            logger.info("Retention report generated")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate retention report: {str(e)}")
            return {"error": str(e)}
    
    def get_retention_statistics(self) -> Dict[str, Any]:
        """Get retention policy statistics"""
        try:
            total_items = len(self.data_items)
            total_policies = len(self.retention_policies)
            total_actions = len(self.retention_actions)
            
            # Count by policy
            policy_counts = {}
            for data_item in self.data_items.values():
                if data_item.retention_policy_id:
                    policy_id = data_item.retention_policy_id
                    policy_counts[policy_id] = policy_counts.get(policy_id, 0) + 1
            
            return {
                "total_items": total_items,
                "total_policies": total_policies,
                "total_actions": total_actions,
                "items_by_policy": policy_counts,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get retention statistics: {str(e)}")
            return {"error": str(e)}
