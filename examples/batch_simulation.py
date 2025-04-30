import numpy as np
from src.disease_model import DiseaseModel, DiseaseParameters
from src.generate_network import generate_social_network
from src.batch_simulation import run_batch_simulation
from src.visualize_network import create_heatmap

def run_batch_simulation_example():
    """Run a batch simulation with varying parameters."""
    # Define parameter ranges
    param_ranges = {
        'transmission_rate': np.linspace(0.2, 0.6, 5),
        'recovery_rate': np.linspace(0.05, 0.2, 4),
        'mortality_rate': np.linspace(0.01, 0.05, 3)
    }
    
    # Define fixed parameters
    fixed_params = {
        'latency_period': 5,
        'immunity_duration': 100,
        'healthcare_capacity': 30
    }
    
    # Define network parameters
    network_params = {
        'n': 200,
        'model': 'watts_strogatz',
        'k': 4,
        'p': 0.1
    }
    
    # Run batch simulation
    results = run_batch_simulation(
        param_ranges=param_ranges,
        fixed_params=fixed_params,
        network_params=network_params,
        n_simulations=3,  # Number of simulations per parameter combination
        n_steps=100,
        disease_model='SIR'
    )
    
    # Print summary statistics
    print("\nBatch Simulation Results:")
    print("-" * 50)
    for param_combo, stats in results.items():
        print(f"\nParameters: {param_combo}")
        print(f"Average Peak Infected: {stats['peak_infected_mean']:.2f} ± {stats['peak_infected_std']:.2f}")
        print(f"Average Total Deaths: {stats['total_deaths_mean']:.2f} ± {stats['total_deaths_std']:.2f}")
        print(f"Average Final Recovered: {stats['final_recovered_mean']:.2f} ± {stats['final_recovered_std']:.2f}")
    
    # Create heatmaps for different metrics
    metrics = ['peak_infected_mean', 'total_deaths_mean', 'final_recovered_mean']
    for metric in metrics:
        # Extract data for heatmap
        x_values = param_ranges['transmission_rate']
        y_values = param_ranges['recovery_rate']
        z_values = np.zeros((len(y_values), len(x_values)))
        
        for i, recovery_rate in enumerate(y_values):
            for j, transmission_rate in enumerate(x_values):
                # Find matching parameter combination
                for param_combo, stats in results.items():
                    if (param_combo['transmission_rate'] == transmission_rate and 
                        param_combo['recovery_rate'] == recovery_rate):
                        z_values[i, j] = stats[metric]
                        break
        
        # Create heatmap
        fig = create_heatmap(
            x_values=x_values,
            y_values=y_values,
            z_values=z_values,
            x_title="Transmission Rate",
            y_title="Recovery Rate",
            title=f"{metric.replace('_', ' ').title()} by Parameter Combination"
        )
        
        fig.show()

if __name__ == "__main__":
    run_batch_simulation_example() 