def initialize_simulation_state(G):
    return {
        'current_step': 0,
        'log': [],
        'graph': G
    }

def step_simulation(sim_state, p_infection, p_recovery, p_death, steps=1, simulate_func=None):
    for _ in range(steps):
        sim_state['graph'], counts = simulate_func(
            sim_state['graph'],
            p_infection,
            p_recovery,
            p_death
        )
        sim_state['current_step'] += 1
        sim_state['log'].append(counts)
    return sim_state
