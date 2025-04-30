import networkx as nx
import matplotlib.pyplot as plt
from src.disease_model import DiseaseModel, DiseaseParameters
from src.generate_network import generate_social_network
from src.visualize_network import create_network_figure, create_time_series_plot

def run_basic_simulation():
    """Run a basic simulation and visualize the results."""
    # Generate a small network
    G = generate_social_network(100, "watts_strogatz")
    
    # Initialize disease parameters
    params = DiseaseParameters(
        transmission_rate=0.3,
        recovery_rate=0.1,
        mortality_rate=0.01,
        latency_period=5,
        immunity_duration=100,
        healthcare_capacity=20
    )
    
    # Create disease model
    model = DiseaseModel(G, "SIR", params)
    
    # Run simulation
    history = []
    for _ in range(50):
        state_counts = model.step()
        history.append(state_counts)
    
    # Visualize results
    print("\nSimulation Results:")
    print("-" * 50)
    print(f"Final State Counts:")
    print(f"Susceptible: {history[-1]['S']}")
    print(f"Infected: {history[-1]['I']}")
    print(f"Recovered: {history[-1]['R']}")
    print(f"Deceased: {history[-1]['D']}")
    
    # Create visualizations
    fig = create_network_figure(model.graph, model.states)
    fig.show()
    
    time_series = create_time_series_plot(history, "Basic SIR Simulation")
    time_series.show()

if __name__ == "__main__":
    run_basic_simulation() 