import random

def simulate_disease_spread(G, p_infection=0.1, steps=10):
    """
    Simulates the disease spread over multiple time steps.
    """
    for _ in range(steps):
        new_infections = []
        for node in G.nodes:
            if G.nodes[node]['status'] == 'infected':
                for neighbor in G.neighbors(node):
                    if G.nodes[neighbor]['status'] == 'healthy':
                        if random.random() < p_infection:
                            new_infections.append(neighbor)
        for node in new_infections:
            G.nodes[node]['status'] = 'infected'
    return G
