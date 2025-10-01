"""
Maintenance Optimizer
Optimizes maintenance scheduling and resource allocation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy.optimize import minimize
import pulp
import json

logger = logging.getLogger(__name__)

@dataclass
class MaintenanceTask:
    """Represents a maintenance task"""
    task_id: str
    equipment_id: str
    task_type: str
    priority: int  # 1-5, 5 being highest
    estimated_duration: int  # hours
    required_skills: List[str]
    cost: float
    failure_risk: float  # 0-1
    due_date: Optional[datetime] = None
    dependencies: List[str] = None

@dataclass
class MaintenanceResource:
    """Represents a maintenance resource (technician, tool, etc.)"""
    resource_id: str
    resource_type: str
    skills: List[str]
    availability: Dict[str, Tuple[int, int]]  # day -> (start_hour, end_hour)
    cost_per_hour: float
    efficiency: float = 1.0

class MaintenanceOptimizer:
    """
    Optimizes maintenance scheduling and resource allocation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.tasks = []
        self.resources = []
        self.schedule = {}
        self.optimization_results = {}
        
        logger.info("MaintenanceOptimizer initialized")
    
    def add_task(self, task: MaintenanceTask):
        """Add a maintenance task"""
        if task.dependencies is None:
            task.dependencies = []
        
        self.tasks.append(task)
        logger.info(f"Added maintenance task: {task.task_id}")
    
    def add_resource(self, resource: MaintenanceResource):
        """Add a maintenance resource"""
        self.resources.append(resource)
        logger.info(f"Added maintenance resource: {resource.resource_id}")
    
    def optimize_schedule(self, 
                         optimization_horizon_days: int = 30,
                         objective: str = 'cost') -> Dict[str, Any]:
        """Optimize maintenance schedule"""
        
        if not self.tasks or not self.resources:
            return {"error": "No tasks or resources available for optimization"}
        
        try:
            if objective == 'cost':
                return self._optimize_cost(optimization_horizon_days)
            elif objective == 'risk':
                return self._optimize_risk(optimization_horizon_days)
            elif objective == 'balanced':
                return self._optimize_balanced(optimization_horizon_days)
            else:
                return {"error": f"Unknown objective: {objective}"}
                
        except Exception as e:
            logger.error(f"Schedule optimization failed: {str(e)}")
            return {"error": f"Optimization failed: {str(e)}"}
    
    def _optimize_cost(self, horizon_days: int) -> Dict[str, Any]:
        """Optimize for minimum cost"""
        
        # Create optimization problem
        prob = pulp.LpProblem("Maintenance_Scheduling_Cost", pulp.LpMinimize)
        
        # Decision variables: x[task_id][day][hour][resource_id] = 1 if task assigned
        x = {}
        for task in self.tasks:
            x[task.task_id] = {}
            for day in range(horizon_days):
                x[task.task_id][day] = {}
                for hour in range(24):
                    x[task.task_id][day][hour] = {}
                    for resource in self.resources:
                        if self._can_assign_task(task, resource, day, hour):
                            x[task.task_id][day][hour][resource.resource_id] = pulp.LpVariable(
                                f"x_{task.task_id}_{day}_{hour}_{resource.resource_id}",
                                cat='Binary'
                            )
        
        # Objective: Minimize total cost
        total_cost = 0
        for task in self.tasks:
            for day in range(horizon_days):
                for hour in range(24):
                    for resource in self.resources:
                        if (task.task_id in x and 
                            day in x[task.task_id] and 
                            hour in x[task.task_id][day] and 
                            resource.resource_id in x[task.task_id][day][hour]):
                            
                            cost = task.cost + (resource.cost_per_hour * task.estimated_duration)
                            total_cost += cost * x[task.task_id][day][hour][resource.resource_id]
        
        prob += total_cost
        
        # Constraints
        self._add_constraints(prob, x, horizon_days)
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status != pulp.LpStatusOptimal:
            return {"error": "Optimization failed to find optimal solution"}
        
        # Extract solution
        schedule = self._extract_solution(x, horizon_days)
        
        # Calculate metrics
        total_cost = pulp.value(prob.objective)
        scheduled_tasks = len([t for t in schedule.values() if t])
        total_tasks = len(self.tasks)
        
        return {
            'objective': 'cost',
            'total_cost': total_cost,
            'scheduled_tasks': scheduled_tasks,
            'total_tasks': total_tasks,
            'scheduling_rate': scheduled_tasks / total_tasks if total_tasks > 0 else 0,
            'schedule': schedule,
            'optimization_status': 'optimal'
        }
    
    def _optimize_risk(self, horizon_days: int) -> Dict[str, Any]:
        """Optimize for minimum risk"""
        
        # Create optimization problem
        prob = pulp.LpProblem("Maintenance_Scheduling_Risk", pulp.LpMinimize)
        
        # Decision variables
        x = {}
        for task in self.tasks:
            x[task.task_id] = {}
            for day in range(horizon_days):
                x[task.task_id][day] = {}
                for hour in range(24):
                    x[task.task_id][day][hour] = {}
                    for resource in self.resources:
                        if self._can_assign_task(task, resource, day, hour):
                            x[task.task_id][day][hour][resource.resource_id] = pulp.LpVariable(
                                f"x_{task.task_id}_{day}_{hour}_{resource.resource_id}",
                                cat='Binary'
                            )
        
        # Objective: Minimize total risk (weighted by failure risk and priority)
        total_risk = 0
        for task in self.tasks:
            for day in range(horizon_days):
                for hour in range(24):
                    for resource in self.resources:
                        if (task.task_id in x and 
                            day in x[task.task_id] and 
                            hour in x[task.task_id][day] and 
                            resource.resource_id in x[task.task_id][day][hour]):
                            
                            # Risk increases with time delay and failure probability
                            time_delay = day
                            risk_weight = task.failure_risk * task.priority * (1 + time_delay * 0.1)
                            total_risk += risk_weight * x[task.task_id][day][hour][resource.resource_id]
        
        prob += total_risk
        
        # Constraints
        self._add_constraints(prob, x, horizon_days)
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status != pulp.LpStatusOptimal:
            return {"error": "Optimization failed to find optimal solution"}
        
        # Extract solution
        schedule = self._extract_solution(x, horizon_days)
        
        # Calculate metrics
        total_risk = pulp.value(prob.objective)
        scheduled_tasks = len([t for t in schedule.values() if t])
        total_tasks = len(self.tasks)
        
        return {
            'objective': 'risk',
            'total_risk': total_risk,
            'scheduled_tasks': scheduled_tasks,
            'total_tasks': total_tasks,
            'scheduling_rate': scheduled_tasks / total_tasks if total_tasks > 0 else 0,
            'schedule': schedule,
            'optimization_status': 'optimal'
        }
    
    def _optimize_balanced(self, horizon_days: int) -> Dict[str, Any]:
        """Optimize for balanced cost and risk"""
        
        # Create optimization problem
        prob = pulp.LpProblem("Maintenance_Scheduling_Balanced", pulp.LpMinimize)
        
        # Decision variables
        x = {}
        for task in self.tasks:
            x[task.task_id] = {}
            for day in range(horizon_days):
                x[task.task_id][day] = {}
                for hour in range(24):
                    x[task.task_id][day][hour] = {}
                    for resource in self.resources:
                        if self._can_assign_task(task, resource, day, hour):
                            x[task.task_id][day][hour][resource.resource_id] = pulp.LpVariable(
                                f"x_{task.task_id}_{day}_{hour}_{resource.resource_id}",
                                cat='Binary'
                            )
        
        # Objective: Balanced cost and risk
        cost_weight = self.config.get('cost_weight', 0.5)
        risk_weight = self.config.get('risk_weight', 0.5)
        
        total_cost = 0
        total_risk = 0
        
        for task in self.tasks:
            for day in range(horizon_days):
                for hour in range(24):
                    for resource in self.resources:
                        if (task.task_id in x and 
                            day in x[task.task_id] and 
                            hour in x[task.task_id][day] and 
                            resource.resource_id in x[task.task_id][day][hour]):
                            
                            # Cost component
                            cost = task.cost + (resource.cost_per_hour * task.estimated_duration)
                            total_cost += cost * x[task.task_id][day][hour][resource.resource_id]
                            
                            # Risk component
                            time_delay = day
                            risk_weight_task = task.failure_risk * task.priority * (1 + time_delay * 0.1)
                            total_risk += risk_weight_task * x[task.task_id][day][hour][resource.resource_id]
        
        prob += cost_weight * total_cost + risk_weight * total_risk
        
        # Constraints
        self._add_constraints(prob, x, horizon_days)
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status != pulp.LpStatusOptimal:
            return {"error": "Optimization failed to find optimal solution"}
        
        # Extract solution
        schedule = self._extract_solution(x, horizon_days)
        
        # Calculate metrics
        total_cost_value = pulp.value(total_cost)
        total_risk_value = pulp.value(total_risk)
        scheduled_tasks = len([t for t in schedule.values() if t])
        total_tasks = len(self.tasks)
        
        return {
            'objective': 'balanced',
            'total_cost': total_cost_value,
            'total_risk': total_risk_value,
            'scheduled_tasks': scheduled_tasks,
            'total_tasks': total_tasks,
            'scheduling_rate': scheduled_tasks / total_tasks if total_tasks > 0 else 0,
            'schedule': schedule,
            'optimization_status': 'optimal'
        }
    
    def _can_assign_task(self, task: MaintenanceTask, resource: MaintenanceResource, 
                        day: int, hour: int) -> bool:
        """Check if a task can be assigned to a resource at a specific time"""
        
        # Check if resource has required skills
        if not all(skill in resource.skills for skill in task.required_skills):
            return False
        
        # Check if resource is available at this time
        day_key = str(day)
        if day_key not in resource.availability:
            return False
        
        start_hour, end_hour = resource.availability[day_key]
        if not (start_hour <= hour < end_hour):
            return False
        
        # Check if task duration fits within resource availability
        if hour + task.estimated_duration > end_hour:
            return False
        
        return True
    
    def _add_constraints(self, prob: pulp.LpProblem, x: Dict, horizon_days: int):
        """Add constraints to the optimization problem"""
        
        # Each task must be assigned at most once
        for task in self.tasks:
            task_assignments = []
            for day in range(horizon_days):
                for hour in range(24):
                    for resource in self.resources:
                        if (task.task_id in x and 
                            day in x[task.task_id] and 
                            hour in x[task.task_id][day] and 
                            resource.resource_id in x[task.task_id][day][hour]):
                            task_assignments.append(x[task.task_id][day][hour][resource.resource_id])
            
            if task_assignments:
                prob += pulp.lpSum(task_assignments) <= 1
        
        # Resource capacity constraints
        for resource in self.resources:
            for day in range(horizon_days):
                for hour in range(24):
                    resource_assignments = []
                    for task in self.tasks:
                        if (task.task_id in x and 
                            day in x[task.task_id] and 
                            hour in x[task.task_id][day] and 
                            resource.resource_id in x[task.task_id][day][hour]):
                            resource_assignments.append(x[task.task_id][day][hour][resource.resource_id])
                    
                    if resource_assignments:
                        prob += pulp.lpSum(resource_assignments) <= 1
        
        # Task dependency constraints
        for task in self.tasks:
            if task.dependencies:
                # Find when dependency tasks are scheduled
                for dep_task_id in task.dependencies:
                    dep_assignments = []
                    for day in range(horizon_days):
                        for hour in range(24):
                            for resource in self.resources:
                                if (dep_task_id in x and 
                                    day in x[dep_task_id] and 
                                    hour in x[dep_task_id][day] and 
                                    resource.resource_id in x[dep_task_id][day][hour]):
                                    dep_assignments.append(x[dep_task_id][day][hour][resource.resource_id])
                    
                    if dep_assignments:
                        # Current task can only be scheduled after dependency
                        for day in range(horizon_days):
                            for hour in range(24):
                                for resource in self.resources:
                                    if (task.task_id in x and 
                                        day in x[task.task_id] and 
                                        hour in x[task.task_id][day] and 
                                        resource.resource_id in x[task.task_id][day][hour]):
                                        
                                        # Sum of dependency assignments up to current time
                                        dep_sum = pulp.lpSum([
                                            x[dep_task_id][d][h][r] 
                                            for d in range(day + 1) 
                                            for h in range(24) 
                                            for r in self.resources
                                            if (dep_task_id in x and 
                                                d in x[dep_task_id] and 
                                                h in x[dep_task_id][d] and 
                                                r in x[dep_task_id][d][h])
                                        ])
                                        
                                        prob += x[task.task_id][day][hour][resource.resource_id] <= dep_sum
    
    def _extract_solution(self, x: Dict, horizon_days: int) -> Dict[str, Any]:
        """Extract solution from optimization results"""
        schedule = {}
        
        for task in self.tasks:
            task_schedule = None
            
            for day in range(horizon_days):
                for hour in range(24):
                    for resource in self.resources:
                        if (task.task_id in x and 
                            day in x[task.task_id] and 
                            hour in x[task.task_id][day] and 
                            resource.resource_id in x[task.task_id][day][hour]):
                            
                            if pulp.value(x[task.task_id][day][hour][resource.resource_id]) == 1:
                                task_schedule = {
                                    'task_id': task.task_id,
                                    'equipment_id': task.equipment_id,
                                    'task_type': task.task_type,
                                    'priority': task.priority,
                                    'scheduled_day': day,
                                    'scheduled_hour': hour,
                                    'duration': task.estimated_duration,
                                    'resource_id': resource.resource_id,
                                    'cost': task.cost + (resource.cost_per_hour * task.estimated_duration),
                                    'failure_risk': task.failure_risk
                                }
                                break
                    
                    if task_schedule:
                        break
                
                if task_schedule:
                    break
            
            schedule[task.task_id] = task_schedule
        
        return schedule
    
    def get_schedule_summary(self) -> Dict[str, Any]:
        """Get summary of the current schedule"""
        if not self.schedule:
            return {"message": "No schedule available"}
        
        scheduled_tasks = [task for task in self.schedule.values() if task]
        total_tasks = len(self.tasks)
        
        # Calculate metrics
        total_cost = sum(task['cost'] for task in scheduled_tasks)
        total_risk = sum(task['failure_risk'] for task in scheduled_tasks)
        
        # Priority distribution
        priority_dist = {}
        for task in scheduled_tasks:
            priority = task['priority']
            priority_dist[priority] = priority_dist.get(priority, 0) + 1
        
        # Resource utilization
        resource_utilization = {}
        for task in scheduled_tasks:
            resource_id = task['resource_id']
            resource_utilization[resource_id] = resource_utilization.get(resource_id, 0) + task['duration']
        
        return {
            'total_tasks': total_tasks,
            'scheduled_tasks': len(scheduled_tasks),
            'scheduling_rate': len(scheduled_tasks) / total_tasks if total_tasks > 0 else 0,
            'total_cost': total_cost,
            'total_risk': total_risk,
            'priority_distribution': priority_dist,
            'resource_utilization': resource_utilization,
            'average_cost_per_task': total_cost / len(scheduled_tasks) if scheduled_tasks else 0
        }
    
    def export_schedule(self, filepath: str):
        """Export schedule to file"""
        schedule_data = {
            'timestamp': datetime.now().isoformat(),
            'schedule': self.schedule,
            'summary': self.get_schedule_summary(),
            'tasks': [task.__dict__ for task in self.tasks],
            'resources': [resource.__dict__ for resource in self.resources]
        }
        
        with open(filepath, 'w') as f:
            json.dump(schedule_data, f, indent=2, default=str)
        
        logger.info(f"Maintenance schedule exported to {filepath}")
    
    def clear_schedule(self):
        """Clear current schedule"""
        self.schedule = {}
        self.tasks = []
        self.resources = []
        logger.info("Maintenance schedule cleared")
