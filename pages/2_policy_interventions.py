import streamlit as st
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from src.disease_model import DiseaseModel, DiseaseParameters, DiseaseState
from src.generate_network import generate_social_network
from src.policy_engine import PolicyEngine, Policy, PolicyTrigger
from src.visualize_network import create_time_series_plot

# Page Config must be the first Streamlit command
st.set_page_config(
    page_title="Policy Interventions",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling. Since there is a default color scheme for light, dark and system. I shall not change the color scheme anymore
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Navbar styling */
    .navbar {
        padding: 1rem;
        margin-bottom: 2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .navbar-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
            text-align:center;
    }
    
    .navbar-links {
        display: flex;
        gap: 1rem;
        padding: 1rem;
        font-size: 1rem;
        justify-content: center;
    }
    
    .navbar-link {
        color: #ADD8E6 !important;
        text-decoration: none !important;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        transition: background-color 0.3s;
    }
    
    .navbar-link:hover {
        background-color: #efefef;
        text-decoration: underline !important;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .stButton>button:hover {
        color: #efefef;
    }
    
    /* Card styling */
    .card {
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Metric styling */
    .metric-card {
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Plot styling */
    .plot-container {
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Expander styling */
    .stExpander {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Navbar
st.markdown("""
<div class="navbar">
    <div class="navbar-title">🦠 Disease Spread Simulation</div>
    <div class="navbar-links">
        <a href="/" target="_self" class="navbar-link">Main Simulation</a>
        <a href="/batch_simulation" target="_self" class="navbar-link">Batch Analysis</a>
        <a href="/policy_interventions" target="_self" class="navbar-link">Policy Interventions</a>
        <a href="/model_comparison" target="_self" class="navbar-link">Model Comparison</a>
    </div>
</div>
""", unsafe_allow_html=True)

st.title("🎯 Policy Interventions Analysis")
st.markdown("""
This page allows you to analyze the effectiveness of different intervention policies
on disease spread, including lockdowns, vaccinations, and healthcare capacity management.
""")

# Network and Disease Parameters
with st.expander("Network and Disease Parameters", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Network Settings")
        num_nodes = st.slider(
            "Population Size",
            100, 2000, 500,
            help="Number of individuals in the population"
        )
        network_model = st.selectbox(
            "Network Model",
            ["watts_strogatz", "erdos_renyi", "barabasi_albert"],
            help="Type of social network structure"
        )
    
    with col2:
        st.markdown("### Disease Parameters")
        transmission_rate = st.slider(
            "Transmission Rate (β)",
            0.0, 1.0, 0.3, 0.01,
            help="Probability of disease transmission per contact"
        )
        recovery_rate = st.slider(
            "Recovery Rate (γ)",
            0.0, 1.0, 0.1, 0.01,
            help="Probability of recovery per timestep"
        )
        mortality_rate = st.slider(
            "Mortality Rate (δ)",
            0.0, 1.0, 0.01, 0.005,
            help="Probability of death instead of recovery"
        )

# Policy Settings
with st.expander("Intervention Settings", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Lockdown Policy")
        lockdown_threshold = st.slider(
            "Infection Rate Threshold",
            0.0, 1.0, 0.2, 0.01,
            help="Infection rate at which lockdown measures are implemented"
        )
        lockdown_strength = st.slider(
            "Lockdown Strength",
            0.0, 1.0, 0.3, 0.05,
            help="Percentage of social connections to remove during lockdown"
        )
    
    with col2:
        st.markdown("### Vaccination Policy")
        vaccination_coverage = st.slider(
            "Vaccination Coverage",
            0.0, 1.0, 0.0, 0.05,
            help="Percentage of population to vaccinate"
        )
        vaccination_strategy = st.selectbox(
            "Vaccination Strategy",
            ["random", "targeted"],
            help="Random: Uniform distribution, Targeted: Focus on high-degree nodes"
        )
    
    with col3:
        st.markdown("### Healthcare Policy")
        healthcare_capacity = st.slider(
            "Healthcare Capacity",
            10, 500, 100, 10,
            help="Maximum number of simultaneous infections before healthcare system overwhelmed"
        )
        healthcare_effectiveness = st.slider(
            "Healthcare Effectiveness",
            0.0, 1.0, 0.5, 0.1,
            help="Effectiveness of healthcare system in reducing mortality"
        )

def run_simulation(with_policies=True):
    """Run a single simulation with or without policies."""
    # Generate network
    G = generate_social_network(num_nodes, network_model)
    
    # Initialize disease parameters
    params = DiseaseParameters(
        transmission_rate=transmission_rate,
        recovery_rate=recovery_rate,
        mortality_rate=mortality_rate,
        latency_period=5,
        immunity_duration=100,
        healthcare_capacity=healthcare_capacity
    )
    
    # Create disease model
    model = DiseaseModel(G, "SIR", params)
    
    # Seed initial infections (5% of population)
    num_initial_infections = int(num_nodes * 0.05)
    nodes_to_infect = np.random.choice(
        list(G.nodes()),
        size=num_initial_infections,
        replace=False
    )
    for node in nodes_to_infect:
        model.states[node] = DiseaseState.INFECTED
        model.time_in_state[node] = 0
    
    # Initialize policy engine if using policies
    policy_engine = None
    if with_policies:
        policy_engine = PolicyEngine()
        
        # Add lockdown policy
        policy_engine.add_policy(Policy(
            name="Lockdown",
            trigger=PolicyTrigger.INFECTION_THRESHOLD,
            threshold=lockdown_threshold,
            action=lambda: model.apply_lockdown(lockdown_strength),
            params={}
        ))
        
        # Add vaccination policy if coverage > 0
        if vaccination_coverage > 0:
            policy_engine.add_policy(Policy(
                name="Vaccination",
                trigger=PolicyTrigger.INFECTION_THRESHOLD,
                threshold=0.0,  # Immediate vaccination
                action=lambda: model.apply_vaccination(
                    vaccination_coverage,
                    vaccination_strategy
                ),
                params={}
            ))
    
    # Run simulation
    history = []
    policy_activations = []
    
    for step in range(100):  # Run for 100 steps
        state_counts = model.step()
        history.append(state_counts)
        
        if with_policies and policy_engine:
            active_policies = policy_engine.evaluate_policies(state_counts)
            if active_policies:
                policy_activations.append({
                    'step': step,
                    'policies': [p.name for p in active_policies]
                })
    
    return history, policy_activations

if st.button("Run Policy Analysis", key="run_analysis"):
    with st.spinner("Running policy analysis..."):
        # Run simulations
        history_with_policies, policy_activations = run_simulation(with_policies=True)
        history_without_policies, _ = run_simulation(with_policies=False)
        
        # Display Results
        st.subheader("Policy Analysis Results")
        
        # Create metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### With Policies")
            with_policies_df = pd.DataFrame(history_with_policies)
            st.metric("Peak Infected", with_policies_df['I'].max())
            st.metric("Total Deaths", with_policies_df['D'].max())
            st.metric("Final Recovered", with_policies_df['R'].iloc[-1])
            st.metric("Epidemic Duration", len(with_policies_df))
        
        with col2:
            st.markdown("### Without Policies")
            without_policies_df = pd.DataFrame(history_without_policies)
            st.metric("Peak Infected", without_policies_df['I'].max())
            st.metric("Total Deaths", without_policies_df['D'].max())
            st.metric("Final Recovered", without_policies_df['R'].iloc[-1])
            st.metric("Epidemic Duration", len(without_policies_df))
        
        # Display policy activations
        if policy_activations:
            st.subheader("Policy Activations")
            activation_df = pd.DataFrame(policy_activations)
            st.dataframe(activation_df)
            
            # Create policy activation timeline
            fig = go.Figure()
            for policy in set([p for act in policy_activations for p in act['policies']]):
                steps = [act['step'] for act in policy_activations if policy in act['policies']]
                fig.add_trace(go.Scatter(
                    x=steps,
                    y=[1] * len(steps),
                    mode='markers',
                    name=policy,
                    marker=dict(size=10)
                ))
            
            fig.update_layout(
                title="Policy Activation Timeline",
                xaxis_title="Time Step",
                yaxis_title="Policy",
                yaxis=dict(showticklabels=False),
                showlegend=True,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Create comparison plot
        st.subheader("Disease Spread Comparison")
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
                x=list(range(len(history_without_policies))),
                y=[h[state] for h in history_without_policies],
                name=f"{state} (Without Policies)",
                line=dict(dash="dash")
            ))
        
        fig.update_layout(
            title="Policy Intervention Comparison",
            xaxis_title="Time Step",
            yaxis_title="Population Count",
            hovermode="x unified",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display policy effectiveness metrics
        st.subheader("Policy Effectiveness")
        
        # Calculate effectiveness metrics
        effectiveness = {
            'Peak Reduction': (without_policies_df['I'].max() - with_policies_df['I'].max()) / without_policies_df['I'].max() * 100,
            'Death Reduction': (without_policies_df['D'].max() - with_policies_df['D'].max()) / without_policies_df['D'].max() * 100,
            'Duration Reduction': (len(without_policies_df) - len(with_policies_df)) / len(without_policies_df) * 100
        }
        
        # Display effectiveness metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Peak Reduction", f"{effectiveness['Peak Reduction']:.1f}%")
        with col2:
            st.metric("Death Reduction", f"{effectiveness['Death Reduction']:.1f}%")
        with col3:
            st.metric("Duration Reduction", f"{effectiveness['Duration Reduction']:.1f}%") 