import networkx as nx
import numpy as np
from typing import Dict, Any

def generate_social_network(num_nodes: int, model_type: str, **kwargs) -> nx.Graph:
    """
    Generate a social network using specified model type and parameters.
    
    Args:
        num_nodes: Number of nodes in the network
        model_type: Type of network model to generate
        **kwargs: Additional parameters for specific network models
        
    Returns:
        networkx.Graph: Generated social network
    """
    if model_type == "watts_strogatz":
        # Default parameters for Watts-Strogatz
        k = kwargs.get('k', 4)  # Each node is connected to k nearest neighbors
        p = kwargs.get('p', 0.1)  # Rewiring probability
        
        G = nx.watts_strogatz_graph(num_nodes, k, p)
        
    elif model_type == "erdos_renyi":
        # Default parameters for Erdős-Rényi
        p = kwargs.get('p', 0.1)  # Probability of edge creation
        
        G = nx.erdos_renyi_graph(num_nodes, p)
        
    elif model_type == "barabasi_albert":
        # Default parameters for Barabási-Albert
        m = kwargs.get('m', 2)  # Number of edges to attach from a new node
        
        G = nx.barabasi_albert_graph(num_nodes, m)
        
    else:
        raise ValueError(f"Unknown network model type: {model_type}")
    
    # Add node attributes
    for node in G.nodes():
        G.nodes[node]['state'] = 'S'  # Initial state: Susceptible
        G.nodes[node]['time_in_state'] = 0
    
    # Add edge weights based on connection strength
    for u, v in G.edges():
        G.edges[u, v]['weight'] = np.random.uniform(0.5, 1.0)
    
    return G

def get_network_parameters(model_type: str) -> Dict[str, Any]:
    """
    Get default parameters for a specific network model.
    
    Args:
        model_type: Type of network model
        
    Returns:
        Dict[str, Any]: Dictionary of parameter names and their default values
    """
    if model_type == "watts_strogatz":
        return {
            'k': 4,  # Each node is connected to k nearest neighbors
            'p': 0.1  # Rewiring probability
        }
    elif model_type == "erdos_renyi":
        return {
            'p': 0.1  # Probability of edge creation
        }
    elif model_type == "barabasi_albert":
        return {
            'm': 2  # Number of edges to attach from a new node
        }
    else:
        raise ValueError(f"Unknown network model type: {model_type}")

def calculate_network_metrics(G: nx.Graph) -> Dict[str, float]:
    """
    Calculate important network metrics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dict[str, float]: Dictionary of network metrics
    """
    metrics = {
        'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'clustering_coefficient': nx.average_clustering(G),
        'average_path_length': nx.average_shortest_path_length(G),
        'diameter': nx.diameter(G),
        'density': nx.density(G),
        'assortativity': nx.degree_assortativity_coefficient(G)
    }
    
    # Calculate community structure using Louvain method
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G)
        metrics['modularity'] = community_louvain.modularity(partition, G)
        metrics['num_communities'] = len(set(partition.values()))
    except ImportError:
        metrics['modularity'] = None
        metrics['num_communities'] = None
    
    return metrics
