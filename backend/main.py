from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import sys
import os
import json
from enum import Enum
import traceback
import networkx as nx
from functools import lru_cache
from time import time
import concurrent.futures
import numpy as np

# Import path handling for your application structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

class DiseaseState(str, Enum):
        SUSCEPTIBLE = "S"
        EXPOSED = "E"
        INFECTED = "I"
        RECOVERED = "R"
        DEAD = "D"

        
class DiseaseParameters:
        def __init__(self, transmission_rate, recovery_rate, 
        mortality_rate, 
                    latency_period, immunity_duration, 
                    healthcare_capacity):
            self.transmission_rate = transmission_rate
            self.recovery_rate = recovery_rate
            self.mortality_rate = mortality_rate
            self.latency_period = latency_period
            self.immunity_duration = immunity_duration
            self.healthcare_capacity = healthcare_capacity
    
class DiseaseModel:
        def __init__(self, G, model_type, disease_params):
            self.G = G
            self.model_type = model_type
            self.params = disease_params
            self.states = {n: DiseaseState.SUSCEPTIBLE for n in G.nodes
            ()}
            self.time_in_state = {n: 0 for n in G.nodes()}
        
        def step(self):
            # Mock step function
            import random
            counts = {
                "S": len([n for n in self.states if self.states[n] == 
                DiseaseState.SUSCEPTIBLE]),
                "E": len([n for n in self.states if self.states[n] == 
                DiseaseState.EXPOSED]),
                "I": len([n for n in self.states if self.states[n] == 
                DiseaseState.INFECTED]),
                "R": len([n for n in self.states if self.states[n] == 
                DiseaseState.RECOVERED]),
                "D": len([n for n in self.states if self.states[n] == 
                DiseaseState.DEAD]),
            }
            # Update some states randomly for the mock
            nodes = list(self.states.keys())
            if nodes:
                node = random.choice(nodes)
                if self.states[node] == DiseaseState.SUSCEPTIBLE:
                    self.states[node] = DiseaseState.INFECTED
                    self.time_in_state[node] = 0
                elif self.states[node] == DiseaseState.INFECTED:
                    self.states[node] = DiseaseState.RECOVERED
                    self.time_in_state[node] = 0
            return counts
    
def generate_social_network(num_nodes, network_model):
        # Mock network generation
        if network_model == "watts_strogatz":
            return nx.watts_strogatz_graph(num_nodes, 4, 0.3)
        elif network_model == "erdos_renyi":
            return nx.erdos_renyi_graph(num_nodes, 0.1)
        elif network_model == "barabasi_albert":
            return nx.barabasi_albert_graph(num_nodes, 3)
        else:
            # Default
            return nx.watts_strogatz_graph(num_nodes, 4, 0.3)

app = FastAPI()

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class SimulationParams(BaseModel):
    num_nodes: int
    network_model: str
    model_type: str
    transmission_rate: float
    recovery_rate: float
    mortality_rate: float
    latency_period: int = 5
    immunity_duration: int = 100
    healthcare_capacity: int = 100
    initial_infection_rate: float = 0.01
    steps: int = 100

class NodeState(BaseModel):
    id: int
    state: str
    time_in_state: int

class Edge(BaseModel):
    source: int
    target: int
    weight: float

class SimulationResult(BaseModel):
    time_series: List[Dict[str, Any]]
    nodes: List[NodeState]
    edges: List[Edge]
    infection_sources: Dict[str, Any]

class BatchResult(BaseModel):
    runs: List[SimulationResult]
    summary: Dict[str, Any]

class PolicyResult(BaseModel):
    with_policy: SimulationResult
    without_policy: SimulationResult
    policy_activations: Optional[List[Dict[str, Any]]] = None
    effectiveness: Optional[Dict[str, Any]] = None

class CompareResult(BaseModel):
    results: Dict[str, Any]  # model_type -> summary/results


# Custom JSON encoder to handle NumPy types
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Helper to convert NodeState, Edge, and SimulationResult to dicts with only native types

def to_native(val):
    if isinstance(val, dict):
        return {k: to_native(v) for k, v in val.items()}
    elif isinstance(val, (np.integer,)):
        return int(val)
    elif isinstance(val, (np.floating,)):
        return float(val)
    elif isinstance(val, (int, float, str, bool)):
        return val
    elif isinstance(val, list):
        return [to_native(v) for v in val]
    elif hasattr(val, 'dict'):
        d = val.dict()
        return to_native(d)
    elif hasattr(val, '__dict__'):
        d = vars(val)
        return to_native(d)
    else:
        return str(val)


@lru_cache(maxsize=32)
def get_cached_layout(G: nx.Graph, layout: str, **kwargs) -> Dict[int, tuple]:
    """Cache network layouts to avoid recalculating them."""
    start_time = time()
    if layout == "spring":
        pos = nx.spring_layout(G, **kwargs)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G, **kwargs)
    elif layout == "circular":
        pos = nx.circular_layout(G, **kwargs)
    else:
        raise ValueError(f"Unknown layout algorithm: {layout}")
    end_time = time()
    print(f"Layout calculation time: {end_time - start_time} seconds")
    return pos


@app.get("/")
def root():
    return {"message": "Disease Spread Simulation API (FastAPI)"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to catch and log all exceptions"""
    error_msg = f"Unhandled error: {str(exc)}"
    error_trace = traceback.format_exc()
    print(f"ERROR: {error_msg}\n{error_trace}")
    return {"error": error_msg, "trace": error_trace}

def run_simulation(model, steps):
    """Run the simulation and return both time_series (counts) and animation_series (node IDs per state), plus newly infected nodes per step."""
    time_series = []
    animation_series = []
    prev_infected = set()
    for _ in range(steps):
        state_counts = model.step()
        # Counts for time series graph
        counts = {k: int(v) for k, v in state_counts.items() if k in ['S', 'E', 'I', 'R', 'D']}
        time_series.append(counts)
        # Node IDs for animation
        state_to_ids = {'S': [], 'E': [], 'I': [], 'R': [], 'D': []}
        for n, s in model.states.items():
            key = s.value if hasattr(s, 'value') else str(s)
            if key in state_to_ids:
                state_to_ids[key].append(int(n))
        # Track newly infected nodes
        current_infected = set(state_to_ids['I'])
        newly_infected = list(current_infected - prev_infected)
        prev_infected = current_infected
        state_to_ids['newly_infected'] = newly_infected
        animation_series.append(state_to_ids)
    return time_series, animation_series

@app.post("/simulate")
async def simulate(params: SimulationParams, sim_state: dict = Body(default=None)):
    try:
        import networkx as nx  # Ensure nx is always available in this scope
        # If sim_state is provided, restore the simulation from it
        if sim_state:
            # Restore network, states, and time_in_state
            G = nx.node_link_graph(sim_state['graph'])
            disease_params = DiseaseParameters(**sim_state['disease_params'])
            model = DiseaseModel(G, sim_state['model_type'], disease_params)
            model.states = {int(k): DiseaseState(v) for k, v in sim_state['states'].items()}
            model.time_in_state = {int(k): int(v) for k, v in sim_state['time_in_state'].items()}
            time_series = sim_state.get('time_series', [])
            animation_series = sim_state.get('animation_series', [])
            node_positions = sim_state.get('node_positions', None)
        else:
            # Start a new simulation
            num_nodes = int(params.num_nodes)
            network_model = str(params.network_model)
            model_type = str(params.model_type)
            transmission_rate = float(params.transmission_rate)
            recovery_rate = float(params.recovery_rate)
            mortality_rate = float(params.mortality_rate)
            latency_period = int(params.latency_period)
            immunity_duration = int(params.immunity_duration)
            healthcare_capacity = int(params.healthcare_capacity)
            initial_infection_rate = float(params.initial_infection_rate)
            steps = int(params.steps)
            G = generate_social_network(num_nodes, network_model)
            disease_params = DiseaseParameters(
                transmission_rate=transmission_rate,
                recovery_rate=recovery_rate,
                mortality_rate=mortality_rate,
                latency_period=latency_period,
                immunity_duration=immunity_duration,
                healthcare_capacity=healthcare_capacity
            )
            model = DiseaseModel(G, model_type, disease_params)
            # Seed initial infections
            num_initial_infections = int(num_nodes * initial_infection_rate)
            if num_initial_infections > 0:
                nodes_to_infect = np.random.choice(list(G.nodes()), size=min(num_initial_infections, len(G.nodes())), replace=False)
                for node in nodes_to_infect:
                    model.states[node] = DiseaseState.INFECTED
                    model.time_in_state[node] = 0
            time_series = []
            animation_series = []
            node_positions = {int(n): {"x": float(np.random.rand()), "y": float(np.random.rand())} for n in G.nodes()}
        # Run the requested number of steps
        steps = int(params.steps)
        for _ in range(steps):
            state_counts = model.step()
            # Counts for time series graph
            counts = {k: int(v) for k, v in state_counts.items() if k in ['S', 'E', 'I', 'R', 'D']}
            time_series.append(counts)
            # Node IDs for animation
            state_to_ids = {'S': [], 'E': [], 'I': [], 'R': [], 'D': []}
            for n, s in model.states.items():
                key = s.value if hasattr(s, 'value') else str(s)
                if key in state_to_ids:
                    state_to_ids[key].append(int(n))
            # Track newly infected nodes
            current_infected = set(state_to_ids['I'])
            prev_infected = set(animation_series[-1]['I']) if animation_series else set()
            newly_infected = list(current_infected - prev_infected)
            state_to_ids['newly_infected'] = newly_infected
            animation_series.append(state_to_ids)
        # Prepare node/edge data
        nodes = []
        edges = []
        for n in G.nodes():
            nodes.append(NodeState(
                id=int(n),
                state=model.states[n].value if hasattr(model.states[n], "value") else str(model.states[n]),
                time_in_state=int(model.time_in_state[n])
            ))
        for u, v in G.edges():
            weight = float(G[u][v].get("weight", 1.0))
            edges.append(Edge(
                source=int(u),
                target=int(v),
                weight=weight
            ))
        # Prepare persistent sim_state for next step
        sim_state_out = {
            'graph': nx.node_link_data(G),
            'disease_params': vars(model.params),
            'model_type': model.model_type,
            'states': {int(k): v.value if hasattr(v, 'value') else str(v) for k, v in model.states.items()},
            'time_in_state': {int(k): int(v) for k, v in model.time_in_state.items()},
            'time_series': time_series,
            'animation_series': animation_series,
            'node_positions': node_positions,
        }
        result = SimulationResult(
            time_series=time_series,
            nodes=nodes,
            edges=edges,
            infection_sources={}  # Not used here
        )
        response = to_native(result)
        response["node_positions"] = node_positions
        response["animation_series"] = animation_series
        response["sim_state"] = sim_state_out
        return response
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in simulate endpoint: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")
    

@app.post("/batch")
def batch(params: SimulationParams):
    num_runs = getattr(params, 'num_runs', 10)
    def run_one(_):
        G = generate_social_network(params.num_nodes, params.network_model)
        disease_params = DiseaseParameters(
            transmission_rate=params.transmission_rate,
            recovery_rate=params.recovery_rate,
            mortality_rate=params.mortality_rate,
            latency_period=params.latency_period,
            immunity_duration=params.immunity_duration,
            healthcare_capacity=params.healthcare_capacity
        )
        model = DiseaseModel(G, params.model_type, disease_params)
        # Seed initial infections
        num_initial_infections = int(params.num_nodes * params.initial_infection_rate)
        if num_initial_infections > 0:
            nodes_to_infect = np.random.choice(list(G.nodes()), size=min(num_initial_infections, len(G.nodes())), replace=False)
            for node in nodes_to_infect:
                model.states[node] = DiseaseState.INFECTED
                model.time_in_state[node] = 0
        time_series, animation_series = run_simulation(model, params.steps)
        return {"time_series": time_series, "animation_series": animation_series}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        runs = list(executor.map(run_one, range(num_runs)))
    # Compute statistics
    metrics = {
        'peak_infected': [],
        'total_deaths': [],
        'epidemic_duration': [],
        'final_recovered': [],
        'attack_rate': [],
        'case_fatality_rate': [],
        'time_to_peak': [],
    }
    for run in runs:
        ts = run['time_series']
        N = params.num_nodes
        infected = [step.get('I', 0) for step in ts]
        deaths = [step.get('D', 0) for step in ts]
        recovered = [step.get('R', 0) for step in ts]
        peak = max(infected)
        time_to_peak = infected.index(peak) if peak > 0 else 0
        total_deaths = deaths[-1] if deaths else 0
        final_recovered = recovered[-1] if recovered else 0
        attack_rate = (final_recovered + total_deaths) / N if N else 0
        case_fatality_rate = total_deaths / (final_recovered + total_deaths) if (final_recovered + total_deaths) else 0
        duration = next((i for i, v in enumerate(infected[::-1]) if v > 0), 0)
        duration = len(infected) - duration if duration else 0
        metrics['peak_infected'].append(peak)
        metrics['total_deaths'].append(total_deaths)
        metrics['epidemic_duration'].append(duration)
        metrics['final_recovered'].append(final_recovered)
        metrics['attack_rate'].append(attack_rate)
        metrics['case_fatality_rate'].append(case_fatality_rate)
        metrics['time_to_peak'].append(time_to_peak)
    statistics = {}
    for k, v in metrics.items():
        arr = np.array(v)
        statistics[k] = {
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'percentile_25': float(np.percentile(arr, 25)),
            'percentile_75': float(np.percentile(arr, 75)),
            'all_values': v,
        }
    out = {'runs': runs, 'statistics': statistics, 'summary': {k: statistics[k]['mean'] for k in statistics}}
    print("/batch return type:", type(out))
    return out

@app.post("/policy")
def policy(params: SimulationParams):
    def run_policy(apply_policy: bool):
        G = generate_social_network(params.num_nodes, params.network_model)
        disease_params = DiseaseParameters(
            transmission_rate=params.transmission_rate,
            recovery_rate=params.recovery_rate,
            mortality_rate=params.mortality_rate,
            latency_period=params.latency_period,
            immunity_duration=params.immunity_duration,
            healthcare_capacity=params.healthcare_capacity
        )
        model = DiseaseModel(G, params.model_type, disease_params)
        # Apply policy if needed (dummy for now)
        if apply_policy:
            model.params.transmission_rate *= 0.7
        # Seed initial infections
        num_initial_infections = int(params.num_nodes * params.initial_infection_rate)
        if num_initial_infections > 0:
            nodes_to_infect = np.random.choice(list(G.nodes()), size=min(num_initial_infections, len(G.nodes())), replace=False)
            for node in nodes_to_infect:
                model.states[node] = DiseaseState.INFECTED
                model.time_in_state[node] = 0
        time_series, animation_series = run_simulation(model, params.steps)
        print("Here below the time series")
        return {"time_series": time_series, "animation_series": animation_series}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        with_policy, without_policy = executor.map(run_policy, [True, False])
        print("Here below the with policy")
    def get_stats(run):
        ts = run['time_series']
        N = params.num_nodes
        infected = [step.get('I', 0) for step in ts]
        deaths = [step.get('D', 0) for step in ts]
        recovered = [step.get('R', 0) for step in ts]
        peak = max(infected)
        total_deaths = deaths[-1] if deaths else 0
        final_recovered = recovered[-1] if recovered else 0
        attack_rate = (final_recovered + total_deaths) / N if N else 0
        case_fatality_rate = total_deaths / (final_recovered + total_deaths) if (final_recovered + total_deaths) else 0
        return {
            'peak_infected': peak,
            'total_deaths': total_deaths,
            'final_recovered': final_recovered,
            'attack_rate': attack_rate,
            'case_fatality_rate': case_fatality_rate,
        }
    out = {
        'with_policy': with_policy,
        'without_policy': without_policy,
        'with_policy_stats': get_stats(with_policy),
        'without_policy_stats': get_stats(without_policy),
    }
    print("/policy return type:", type(out))
    return out

@app.post("/compare")
def compare(params: SimulationParams):
    model_types = ['SIR', 'SEIR', 'SIRD']
    def run_model(model_type):
        G = generate_social_network(params.num_nodes, params.network_model)
        disease_params = DiseaseParameters(
            transmission_rate=params.transmission_rate,
            recovery_rate=params.recovery_rate,
            mortality_rate=params.mortality_rate,
            latency_period=params.latency_period,
            immunity_duration=params.immunity_duration,
            healthcare_capacity=params.healthcare_capacity
        )
        model = DiseaseModel(G, model_type, disease_params)
        # Seed initial infections
        num_initial_infections = int(params.num_nodes * params.initial_infection_rate)
        if num_initial_infections > 0:
            nodes_to_infect = np.random.choice(list(G.nodes()), size=min(num_initial_infections, len(G.nodes())), replace=False)
            for node in nodes_to_infect:
                model.states[node] = DiseaseState.INFECTED
                model.time_in_state[node] = 0
        time_series, animation_series = run_simulation(model, params.steps)
        return {"time_series": time_series, "animation_series": animation_series}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(run_model, model_types))
    out = {mt: res for mt, res in zip(model_types, results)}
    def get_stats(run):
        ts = run['time_series']
        N = params.num_nodes
        infected = [step.get('I', 0) for step in ts]
        deaths = [step.get('D', 0) for step in ts]
        recovered = [step.get('R', 0) for step in ts]
        peak = max(infected)
        total_deaths = deaths[-1] if deaths else 0
        final_recovered = recovered[-1] if recovered else 0
        attack_rate = (final_recovered + total_deaths) / N if N else 0
        case_fatality_rate = total_deaths / (final_recovered + total_deaths) if (final_recovered + total_deaths) else 0
        return {
            'peak_infected': peak,
            'total_deaths': total_deaths,
            'final_recovered': final_recovered,
            'attack_rate': attack_rate,
            'case_fatality_rate': case_fatality_rate,
        }
    stats = {mt: get_stats(out[mt]) for mt in model_types}
    out = {'results': out, 'stats': stats}
    print("/compare return type:", type(out))
    return out