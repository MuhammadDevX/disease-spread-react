import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class DiseaseState(Enum):
    SUSCEPTIBLE = "S"
    EXPOSED = "E"
    INFECTED = "I"
    RECOVERED = "R"
    DEAD = "D"

@dataclass
class DiseaseParameters:
    transmission_rate: float  # β
    recovery_rate: float      # γ
    mortality_rate: float     # δ
    latency_period: int       # λ (for SEIR)
    immunity_duration: int    # Duration of immunity before becoming susceptible again
    healthcare_capacity: int  # Maximum number of simultaneous infections before healthcare system overwhelmed

class DiseaseModel:
    def __init__(
        self,
        graph: nx.Graph,
        model_type: str = "SIR",
        params: Optional[DiseaseParameters] = None
    ):
        self.graph = graph
        self.model_type = model_type.upper()
        self.params = params or DiseaseParameters(
            transmission_rate=0.3,
            recovery_rate=0.1,
            mortality_rate=0.01,
            latency_period=5,
            immunity_duration=100,
            healthcare_capacity=100
        )
        
        # Initialize node states
        self.states = {node: DiseaseState.SUSCEPTIBLE for node in graph.nodes()}
        self.time_in_state = {node: 0 for node in graph.nodes()}
        self.healthcare_overwhelmed = False
        
    def update_healthcare_status(self, current_infected: int) -> None:
        """Update healthcare system status based on current infected count"""
        self.healthcare_overwhelmed = current_infected > self.params.healthcare_capacity
        
    def get_effective_recovery_rate(self) -> float:
        """Adjust recovery rate based on healthcare system status"""
        if self.healthcare_overwhelmed:
            return self.params.recovery_rate * 0.5  # Reduced recovery rate when overwhelmed
        return self.params.recovery_rate
        
    def get_effective_mortality_rate(self) -> float:
        """Adjust mortality rate based on healthcare system status"""
        if self.healthcare_overwhelmed:
            return self.params.mortality_rate * 2.0  # Increased mortality when overwhelmed
        return self.params.mortality_rate
        
    def step(self) -> Dict[str, int]:
        """Perform one step of the simulation"""
        new_states = self.states.copy()
        current_infected = sum(1 for state in self.states.values() if state == DiseaseState.INFECTED)
        
        self.update_healthcare_status(current_infected)
        effective_recovery_rate = self.get_effective_recovery_rate()
        effective_mortality_rate = self.get_effective_mortality_rate()
        
        # Track infection sources
        infection_sources = {}
        
        # Process each node
        for node in self.graph.nodes():
            current_state = self.states[node]
            self.time_in_state[node] += 1
            
            if current_state == DiseaseState.SUSCEPTIBLE:
                # Check for infection from neighbors
                infected_neighbors = [
                    n for n in self.graph.neighbors(node)
                    if self.states[n] == DiseaseState.INFECTED
                ]
                if infected_neighbors and np.random.random() < self.params.transmission_rate:
                    # Choose a random infected neighbor as the source
                    source = np.random.choice(infected_neighbors)
                    infection_sources[node] = source
                    if self.model_type == "SEIR":
                        new_states[node] = DiseaseState.EXPOSED
                    else:
                        new_states[node] = DiseaseState.INFECTED
                    self.time_in_state[node] = 0
                    
            elif current_state == DiseaseState.EXPOSED:
                if self.time_in_state[node] >= self.params.latency_period:
                    new_states[node] = DiseaseState.INFECTED
                    self.time_in_state[node] = 0
                    
            elif current_state == DiseaseState.INFECTED:
                if np.random.random() < effective_recovery_rate:
                    if np.random.random() < effective_mortality_rate:
                        new_states[node] = DiseaseState.DEAD
                    else:
                        new_states[node] = DiseaseState.RECOVERED
                    self.time_in_state[node] = 0
                    
            elif current_state == DiseaseState.RECOVERED:
                if self.time_in_state[node] >= self.params.immunity_duration:
                    new_states[node] = DiseaseState.SUSCEPTIBLE
                    self.time_in_state[node] = 0
        
        self.states = new_states
        
        # Return current state counts and infection sources
        return {
            "S": sum(1 for state in self.states.values() if state == DiseaseState.SUSCEPTIBLE),
            "E": sum(1 for state in self.states.values() if state == DiseaseState.EXPOSED),
            "I": sum(1 for state in self.states.values() if state == DiseaseState.INFECTED),
            "R": sum(1 for state in self.states.values() if state == DiseaseState.RECOVERED),
            "D": sum(1 for state in self.states.values() if state == DiseaseState.DEAD),
            "infection_sources": infection_sources
        }
        
    def apply_vaccination(self, coverage: float, strategy: str = "random") -> None:
        """Apply vaccination to the population"""
        if strategy == "random":
            nodes_to_vaccinate = np.random.choice(
                list(self.graph.nodes()),
                size=int(len(self.graph) * coverage),
                replace=False
            )
        else:  # targeted vaccination based on degree
            degrees = dict(self.graph.degree())
            nodes_to_vaccinate = sorted(
                degrees.keys(),
                key=lambda x: degrees[x],
                reverse=True
            )[:int(len(self.graph) * coverage)]
            
        for node in nodes_to_vaccinate:
            if self.states[node] == DiseaseState.SUSCEPTIBLE:
                self.states[node] = DiseaseState.RECOVERED
                self.time_in_state[node] = 0
                
    def apply_lockdown(self, threshold: float) -> None:
        """Apply lockdown by removing edges based on infection density"""
        current_infected = sum(1 for state in self.states.values() if state == DiseaseState.INFECTED)
        infection_density = current_infected / len(self.graph)
        
        if infection_density > threshold:
            # Convert edges to list of tuples for proper handling
            edges_list = list(self.graph.edges())
            num_edges_to_remove = int(len(edges_list) * 0.3)  # Remove 30% of edges
            
            if num_edges_to_remove > 0:
                edges_to_remove = np.random.choice(
                    len(edges_list),  # Use indices instead of edges directly
                    size=num_edges_to_remove,
                    replace=False
                )
                edges_to_remove = [edges_list[i] for i in edges_to_remove]
                self.graph.remove_edges_from(edges_to_remove) 