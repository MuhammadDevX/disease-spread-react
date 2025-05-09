import streamlit as st
import networkx as nx
import numpy as np
from src.disease_model import DiseaseModel, DiseaseParameters, DiseaseState
from src.generate_network import generate_social_network
from src.visualize_network import create_time_series_plot
import plotly.graph_objects as go
import pandas as pd

# Page Config must be the first Streamlit command
st.set_page_config(
    page_title="Model Comparison",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
        background-color: #ffffff;
    }
    
    /* Navbar styling */
    .navbar {
        background-color: #ffffff;
        padding: 1rem;
        margin-bottom: 2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .navbar-title {
        color: #1a237e;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .navbar-links {
        display: flex;
        gap: 1rem;
    }
    
    .navbar-link {
        color: #1a237e;
        text-decoration: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        transition: background-color 0.3s;
    }
    
    .navbar-link:hover {
        background-color: #283593;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: #ffffff;
        color: #1a237e;
        padding: 0.5rem 1rem;
        border: 1px solid #1a237e;
        border-radius: 4px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #283593;
        color: #ffffff;
    }
    
    /* Card styling */
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        border: 1px solid #1a237e;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Metric styling */
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Plot styling */
    .plot-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Expander styling */
    .stExpander {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Slider styling */
    .stSlider {
        background-color: #ffffff;
    }
    
    /* Selectbox styling */
    .stSelectbox {
        background-color: #ffffff;
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

st.title("📈 Model Comparison")
st.markdown("""
This page allows you to compare different disease spread models (SIR, SEIR, SIRD)
and network models to understand their impact on disease spread patterns.
""")

# Network Parameters
with st.expander("Network Settings", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
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
        selected_models = st.multiselect(
            "Select Models to Compare",
            ["SIR", "SEIR", "SIRD"],
            default=["SIR", "SEIR"],
            help="Choose which disease models to compare"
        )

# Disease Parameters
with st.expander("Disease Parameters", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
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
    
    with col2:
        mortality_rate = st.slider(
            "Mortality Rate (δ)",
            0.0, 1.0, 0.01, 0.005,
            help="Probability of death instead of recovery"
        )
        latency_period = st.slider(
            "Latency Period (λ)",
            1, 14, 5,
            help="Number of days before becoming infectious"
        )
    
    with col3:
        immunity_duration = st.slider(
            "Immunity Duration",
            10, 365, 100,
            help="Number of days before becoming susceptible again"
        )
        healthcare_capacity = st.slider(
            "Healthcare Capacity",
            10, 500, 100, 10,
            help="Maximum number of simultaneous infections before healthcare system overwhelmed"
        )

def run_simulation(model_type):
    """Run a single simulation for a given model type."""
    # Generate network
    G = generate_social_network(num_nodes, network_model)
    
    # Initialize disease parameters
    params = DiseaseParameters(
        transmission_rate=transmission_rate,
        recovery_rate=recovery_rate,
        mortality_rate=mortality_rate,
        latency_period=latency_period,
        immunity_duration=immunity_duration,
        healthcare_capacity=healthcare_capacity
    )
    
    # Create disease model
    model = DiseaseModel(G, model_type, params)
    
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
    
    # Run simulation
    history = []
    for _ in range(100):  # Run for 100 steps
        state_counts = model.step()
        history.append(state_counts)
    
    return history

if st.button("Run Model Comparison"):
    if not selected_models:
        st.error("Please select at least one model to compare.")
    else:
        results = {}
        
        for model_type in selected_models:
            # Run multiple simulations for each model
            histories = []
            for _ in range(5):  # Run 5 simulations per model
                history = run_simulation(model_type)
                histories.append(history)
            
            # Calculate metrics for each simulation
            metrics_list = []
            for history in histories:
                df = pd.DataFrame(history)
                metrics = {
                    'peak_infected': df['I'].max(),
                    'total_deaths': df['D'].max(),
                    'final_recovered': df['R'].iloc[-1],
                    'epidemic_duration': len(df),
                    'R0': df['I'].sum() / df['R'].sum() if df['R'].sum() > 0 else 0
                }
                metrics_list.append(metrics)
            
            results[model_type] = {
                'histories': histories,
                'metrics': metrics_list
            }
        
        # Display Results
        st.subheader("Model Comparison Results")
        
        # Create comparison metrics table
        comparison_data = []
        for model_type, result in results.items():
            metrics_df = pd.DataFrame(result['metrics'])
            row = {
                'Model': model_type,
                'Mean R0': metrics_df['R0'].mean(),
                'Mean Peak Infected': metrics_df['peak_infected'].mean(),
                'Mean Total Deaths': metrics_df['total_deaths'].mean(),
                'Mean Epidemic Duration': metrics_df['epidemic_duration'].mean()
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df)
        
        # Create comparison plot
        fig = go.Figure()
        
        for model_type, result in results.items():
            # Calculate average history
            avg_history = []
            for step in range(100):
                step_counts = {}
                for state in ['S', 'E', 'I', 'R', 'D']:
                    counts = []
                    for history in result['histories']:
                        if step < len(history):
                            counts.append(history[step].get(state, 0))
                    step_counts[state] = np.mean(counts) if counts else 0
                avg_history.append(step_counts)
            
            # Add traces for each state
            for state in ['S', 'E', 'I', 'R', 'D']:
                if state in avg_history[0]:  # Only add if state exists in model
                    fig.add_trace(go.Scatter(
                        x=list(range(len(avg_history))),
                        y=[h[state] for h in avg_history],
                        name=f"{state} ({model_type})",
                        line=dict(dash="solid" if model_type == "SIR" else "dot" if model_type == "SEIR" else "dash")
                    ))
        
        fig.update_layout(
            title="Model Comparison",
            xaxis_title="Time Step",
            yaxis_title="Population Count",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display model-specific insights
        st.subheader("Model Insights")
        
        for model_type in selected_models:
            st.markdown(f"### {model_type} Model")
            if model_type == "SIR":
                st.markdown("""
                - Basic model with Susceptible, Infected, and Recovered states
                - No consideration of incubation period
                - Assumes immediate infectiousness upon infection
                - Best for diseases with short incubation periods
                """)
            elif model_type == "SEIR":
                st.markdown("""
                - Includes Exposed state for incubation period
                - More realistic representation of disease progression
                - Better for diseases with significant incubation periods
                - Useful for modeling diseases like COVID-19
                """)
            elif model_type == "SIRD":
                st.markdown("""
                - Includes Death state
                - Accounts for disease mortality
                - Useful for highly fatal diseases
                - Helps model healthcare system impact
                """)
                

