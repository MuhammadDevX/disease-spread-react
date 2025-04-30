import networkx as nx
import matplotlib.pyplot as plt
from src.disease_model import DiseaseModel, DiseaseParameters
from src.generate_network import generate_social_network
from src.policy_engine import PolicyEngine, Policy, PolicyTrigger
from src.visualize_network import create_time_series_plot

def run_policy_intervention_simulation():
    """Run a simulation with policy interventions and compare results."""
    # Generate network
    G = generate_social_network(200, "barabasi_albert")
    
    # Initialize disease parameters
    params = DiseaseParameters(
        transmission_rate=0.4,
        recovery_rate=0.1,
        mortality_rate=0.02,
        latency_period=5,
        immunity_duration=100,
        healthcare_capacity=30
    )
    
    # Create disease model
    model = DiseaseModel(G, "SIR", params)
    
    # Initialize policy engine
    policy_engine = PolicyEngine()
    
    # Add policies
    policy_engine.add_policy(Policy(
        name="Lockdown",
        trigger=PolicyTrigger.INFECTION_THRESHOLD,
        threshold=0.2,  # 20% infection rate
        action=lambda: model.apply_lockdown(0.2),
        params={}
    ))
    
    policy_engine.add_policy(Policy(
        name="Vaccination",
        trigger=PolicyTrigger.INFECTION_THRESHOLD,
        threshold=0.1,  # 10% infection rate
        action=lambda: model.apply_vaccination(0.3, "targeted"),  # 30% coverage
        params={}
    ))
    
    # Run simulation with policies
    history_with_policies = []
    for _ in range(100):
        state_counts = model.step()
        history_with_policies.append(state_counts)
        policy_engine.evaluate_policies(state_counts)
    
    # Run simulation without policies (control)
    G_control = generate_social_network(200, "barabasi_albert")
    model_control = DiseaseModel(G_control, "SIR", params)
    history_control = []
    for _ in range(100):
        state_counts = model_control.step()
        history_control.append(state_counts)
    
    # Print results
    print("\nSimulation Results:")
    print("-" * 50)
    print("With Policies:")
    print(f"Peak Infected: {max(h['I'] for h in history_with_policies)}")
    print(f"Total Deaths: {history_with_policies[-1]['D']}")
    print(f"Final Recovered: {history_with_policies[-1]['R']}")
    
    print("\nWithout Policies (Control):")
    print(f"Peak Infected: {max(h['I'] for h in history_control)}")
    print(f"Total Deaths: {history_control[-1]['D']}")
    print(f"Final Recovered: {history_control[-1]['R']}")
    
    # Print policy activations
    print("\nPolicy Activations:")
    print("-" * 50)
    for event in policy_engine.get_policy_history():
        print(f"Step {event['step']}: {event['policy']} activated "
              f"(trigger: {event['trigger']}, value: {event['trigger_value']:.2f})")
    
    # Create comparison plot
    fig = go.Figure()
    
    # Add traces for each state with policies
    for state in ['S', 'I', 'R', 'D']:
        fig.add_trace(go.Scatter(
            x=list(range(len(history_with_policies))),
            y=[h[state] for h in history_with_policies],
            name=f"{state} (With Policies)",
            line=dict(dash="solid")
        ))
    
    # Add traces for each state without policies
    for state in ['S', 'I', 'R', 'D']:
        fig.add_trace(go.Scatter(
            x=list(range(len(history_control))),
            y=[h[state] for h in history_control],
            name=f"{state} (Without Policies)",
            line=dict(dash="dash")
        ))
    
    fig.update_layout(
        title="Policy Intervention Comparison",
        xaxis_title="Time Step",
        yaxis_title="Population Count",
        hovermode="x unified"
    )
    
    fig.show()

if __name__ == "__main__":
    run_policy_intervention_simulation() 