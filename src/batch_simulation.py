import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor
from src.disease_model import DiseaseModel, DiseaseParameters
from src.generate_network import generate_social_network
import networkx as nx

def run_single_simulation(
    num_nodes: int,
    network_model: str,
    model_type: str,
    params: DiseaseParameters,
    num_steps: int
) -> Tuple[List[Dict[str, int]], Dict[str, float]]:
    """
    Run a single simulation and return its history and metrics.
    
    Args:
        num_nodes: Number of nodes in the network
        network_model: Type of network model to use
        model_type: Type of disease model (SIR, SEIR, SIRD)
        params: Disease parameters
        num_steps: Number of simulation steps
        
    Returns:
        Tuple containing:
        - List of state counts at each step
        - Dictionary of simulation metrics
    """
    # Generate network
    G = generate_social_network(num_nodes, network_model)
    
    # Initialize model
    model = DiseaseModel(G, model_type, params)
    
    # Run simulation
    history = []
    for _ in range(num_steps):
        state_counts = model.step()
        history.append(state_counts)
    
    # Calculate metrics
    metrics = calculate_simulation_metrics(history)
    
    return history, metrics

def calculate_simulation_metrics(history: List[Dict[str, int]]) -> Dict[str, float]:
    """
    Calculate key metrics from simulation history.
    
    Args:
        history: List of state counts at each step
        
    Returns:
        Dictionary of simulation metrics
    """
    df = pd.DataFrame(history)
    
    metrics = {
        'peak_infected': df['I'].max(),
        'peak_time': df['I'].idxmax(),
        'total_infected': df['I'].sum(),
        'total_deaths': df['D'].max(),
        'final_recovered': df['R'].iloc[-1],
        'epidemic_duration': len(df),
        'attack_rate': df['I'].sum() / (df['S'].iloc[0] + df['I'].sum()),
        'case_fatality_rate': df['D'].max() / df['I'].sum() if df['I'].sum() > 0 else 0
    }
    
    # Calculate R0 (basic reproduction number)
    if len(df) > 1:
        new_infections = df['I'].diff().fillna(0)
        recovery_rate = df['R'].diff().fillna(0)
        metrics['R0'] = new_infections.sum() / recovery_rate.sum() if recovery_rate.sum() > 0 else 0
    else:
        metrics['R0'] = 0
    
    return metrics

def run_batch_simulations(
    num_simulations: int,
    num_nodes: int,
    network_model: str,
    model_type: str,
    params: DiseaseParameters,
    num_steps: int,
    num_workers: int = 4
) -> Tuple[List[List[Dict[str, int]]], List[Dict[str, float]]]:
    """
    Run multiple simulations in parallel and return their histories and metrics.
    
    Args:
        num_simulations: Number of simulations to run
        num_nodes: Number of nodes in the network
        network_model: Type of network model to use
        model_type: Type of disease model (SIR, SEIR, SIRD)
        params: Disease parameters
        num_steps: Number of simulation steps
        num_workers: Number of parallel workers
        
    Returns:
        Tuple containing:
        - List of simulation histories
        - List of simulation metrics
    """
    histories = []
    metrics_list = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                run_single_simulation,
                num_nodes,
                network_model,
                model_type,
                params,
                num_steps
            )
            for _ in range(num_simulations)
        ]
        
        for future in futures:
            history, metrics = future.result()
            histories.append(history)
            metrics_list.append(metrics)
    
    return histories, metrics_list

def analyze_batch_results(
    histories: List[List[Dict[str, int]]],
    metrics_list: List[Dict[str, float]]
) -> Dict[str, Any]:
    """
    Analyze results from batch simulations.
    
    Args:
        histories: List of simulation histories
        metrics_list: List of simulation metrics
        
    Returns:
        Dictionary containing analysis results
    """
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    # Calculate statistics
    analysis = {
        'mean_metrics': metrics_df.mean().to_dict(),
        'std_metrics': metrics_df.std().to_dict(),
        'median_metrics': metrics_df.median().to_dict(),
        'min_metrics': metrics_df.min().to_dict(),
        'max_metrics': metrics_df.max().to_dict()
    }
    
    # Calculate confidence intervals
    for metric in metrics_df.columns:
        mean = metrics_df[metric].mean()
        std = metrics_df[metric].std()
        n = len(metrics_df)
        se = std / np.sqrt(n)
        analysis[f'{metric}_ci'] = {
            'lower': mean - 1.96 * se,
            'upper': mean + 1.96 * se
        }
    
    # Calculate average time series
    max_steps = max(len(history) for history in histories)
    avg_history = []
    for step in range(max_steps):
        step_counts = {}
        for state in ['S', 'E', 'I', 'R', 'D']:
            counts = []
            for history in histories:
                if step < len(history):
                    counts.append(history[step].get(state, 0))
            step_counts[state] = np.mean(counts) if counts else 0
        avg_history.append(step_counts)
    
    analysis['average_history'] = avg_history
    
    return analysis

def compare_network_models(
    num_nodes: int,
    model_types: List[str],
    params: DiseaseParameters,
    num_steps: int,
    num_simulations: int = 10
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different network models through batch simulations.
    
    Args:
        num_nodes: Number of nodes in the network
        model_types: List of network models to compare
        params: Disease parameters
        num_steps: Number of simulation steps
        num_simulations: Number of simulations per model
        
    Returns:
        Dictionary mapping network models to their analysis results
    """
    results = {}
    
    for model_type in model_types:
        histories, metrics_list = run_batch_simulations(
            num_simulations,
            num_nodes,
            model_type,
            'SIR',  # Using SIR model for comparison
            params,
            num_steps
        )
        
        analysis = analyze_batch_results(histories, metrics_list)
        results[model_type] = analysis
    
    return results 