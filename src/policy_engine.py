from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

class PolicyTrigger(Enum):
    INFECTION_THRESHOLD = "infection_threshold"
    PEAK_INFECTION = "peak_infection"
    REINFECTION_RATE = "reinfection_rate"
    HEALTHCARE_CAPACITY = "healthcare_capacity"

@dataclass
class Policy:
    name: str
    trigger: PolicyTrigger
    threshold: float
    action: Callable
    params: Dict[str, Any]
    active: bool = False

class PolicyEngine:
    def __init__(self):
        self.policies: List[Policy] = []
        self.history: List[Dict] = []
        
    def add_policy(self, policy: Policy) -> None:
        """Add a new policy to the engine"""
        self.policies.append(policy)
        
    def evaluate_policies(self, simulation_state: Dict[str, int]) -> List[Policy]:
        """
        Evaluate all policies based on current simulation state.
        
        Args:
            simulation_state: Dictionary containing counts of each state
            
        Returns:
            List of policies that should be activated
        """
        active_policies = []
        
        # Calculate total population by summing only the integer values
        total_population = sum(v for v in simulation_state.values() if isinstance(v, int))
        
        # Calculate infection rate
        infected_count = simulation_state.get('infected', 0)
        infection_rate = infected_count / total_population if total_population > 0 else 0
        
        # Evaluate each policy
        for policy in self.policies:
            if not policy.active and infection_rate >= policy.threshold:
                policy.active = True
                policy.action(**policy.params)
                active_policies.append(policy)
                
                # Log policy activation
                self.history.append({
                    "step": len(self.history),
                    "policy": policy.name,
                    "trigger": policy.trigger.value,
                    "trigger_value": infection_rate,
                    "threshold": policy.threshold
                })
        
        return active_policies
        
    def _get_trigger_value(self, trigger: PolicyTrigger, state: Dict) -> float:
        """Get the current value for a policy trigger"""
        if trigger == PolicyTrigger.INFECTION_THRESHOLD:
            total_population = sum(state.values())
            return state.get("I", 0) / total_population
            
        elif trigger == PolicyTrigger.PEAK_INFECTION:
            # Calculate if we're approaching peak infection
            if len(self.history) < 2:
                return 0
            prev_infected = self.history[-1].get("I", 0)
            current_infected = state.get("I", 0)
            return (current_infected - prev_infected) / prev_infected if prev_infected > 0 else 0
            
        elif trigger == PolicyTrigger.REINFECTION_RATE:
            # Calculate reinfection rate from history
            if len(self.history) < 2:
                return 0
            prev_recovered = self.history[-1].get("R", 0)
            current_susceptible = state.get("S", 0)
            return (current_susceptible - prev_recovered) / prev_recovered if prev_recovered > 0 else 0
            
        elif trigger == PolicyTrigger.HEALTHCARE_CAPACITY:
            return state.get("I", 0) / state.get("healthcare_capacity", 100)
            
        return 0
        
    def get_policy_history(self) -> List[Dict]:
        """Get the history of policy activations"""
        return self.history
        
    def reset(self) -> None:
        """Reset all policies and history"""
        for policy in self.policies:
            policy.active = False
        self.history = [] 