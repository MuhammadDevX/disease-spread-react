import networkx as nx
import random

def generate_social_network(num_nodes=500, model='watts_strogatz'):
    """
    Generates a synthetic social network.
    """
    G = None
    if model == 'erdos_renyi':
        G = nx.erdos_renyi_graph(num_nodes, 0.05)
    elif model == 'barabasi_albert':
        G = nx.barabasi_albert_graph(num_nodes, 3)
    else:  # watts_strogatz
        G = nx.watts_strogatz_graph(num_nodes, 6, 0.4)

    nx.set_node_attributes(G, 'healthy', 'status')
    initial_infected = random.sample(list(G.nodes), 5)
    for node in initial_infected:
        G.nodes[node]['status'] = 'infected'

    return G
