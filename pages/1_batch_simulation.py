import streamlit as st
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from src.disease_model import DiseaseModel, DiseaseParameters, DiseaseState
from src.generate_network import generate_social_network
from src.visualize_network import create_time_series_plot

# Page Config must be the first Streamlit command
st.set_page_config(
    page_title="Batch Simulation Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
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
        text-align: center;
    }
    
    .navbar-links {
        display: flex;
        gap: 1rem;
        font-size: 1rem;
        justify-content: center;
        padding: 1rem;
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

st.title("📊 Batch Simulation Analysis")
st.markdown("""
This page allows you to run multiple simulations in parallel to analyze disease spread patterns
across different network models and parameters.
""")

# Batch Simulation Parameters
col1, col2 = st.columns(2)

with col1:
    num_simulations = st.slider(
        "Number of Simulations",
        1, 100, 10,
        help="Number of independent simulations to run"
    )
    num_steps = st.slider(
        "Steps per Simulation",
        10, 500, 100,
        help="Number of time steps per simulation"
    )
    num_workers = st.slider(
        "Number of Workers",
        1, 8, 4,
        help="Number of parallel workers for batch processing"
    )

with col2:
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
    model_type = st.selectbox(
        "Disease Model",
        ["SIR", "SEIR", "SIRD"],
        help="Type of disease spread model"
    )

# Disease Parameters
st.subheader("Disease Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    transmission_rate = st.slider(
        "Transmission Rate (β)",
        0.0, 1.0, 0.3, 0.01
    )
with col2:
    recovery_rate = st.slider(
        "Recovery Rate (γ)",
        0.0, 1.0, 0.1, 0.01
    )
with col3:
    mortality_rate = st.slider(
        "Mortality Rate (δ)",
        0.0, 1.0, 0.01, 0.005
    )

# Initialize disease parameters
params = DiseaseParameters(
    transmission_rate=transmission_rate,
    recovery_rate=recovery_rate,
    mortality_rate=mortality_rate,
    latency_period=5,
    immunity_duration=100,
    healthcare_capacity=100
)

def run_single_simulation():
    """Run a single simulation and return its history."""
    # Generate network
    G = generate_social_network(num_nodes, network_model)
    
    # Initialize model
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
    for _ in range(num_steps):
        state_counts = model.step()
        history.append(state_counts)
    
    return history

if st.button("Run Batch Simulations"):
    with st.spinner("Running batch simulations..."):
        # Run simulations
        histories = []
        for i in range(num_simulations):
            history = run_single_simulation()
            histories.append(history)
        
        # Calculate metrics for each simulation
        metrics_list = []
        for history in histories:
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
            metrics_list.append(metrics)
        
        # Display Results
        st.subheader("Simulation Results")
        
        # Metrics Summary
        st.write("### Key Metrics")
        metrics_df = pd.DataFrame(metrics_list)
        summary_df = pd.DataFrame({
            'Mean': metrics_df.mean(),
            'Std Dev': metrics_df.std(),
            'Min': metrics_df.min(),
            'Max': metrics_df.max()
        })
        st.dataframe(summary_df)
        
        # Average Time Series
        st.write("### Average Population States Over Time")
        avg_history = []
        for step in range(num_steps):
            step_counts = {}
            for state in ['S', 'E', 'I', 'R', 'D']:
                counts = []
                for history in histories:
                    if step < len(history):
                        counts.append(history[step].get(state, 0))
                step_counts[state] = np.mean(counts) if counts else 0
            avg_history.append(step_counts)
        
        fig = create_time_series_plot(
            avg_history,
            f"Average {model_type} Model Simulation"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution Plots
        st.write("### Metric Distributions")
        for metric in ['peak_infected', 'total_deaths', 'attack_rate']:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=metrics_df[metric],
                name=metric,
                nbinsx=20
            ))
            fig.update_layout(
                title=f"Distribution of {metric}",
                xaxis_title=metric,
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)

# Network Model Comparison
st.header("Network Model Comparison")

if st.button("Compare Network Models"):
    with st.spinner("Running network model comparison..."):
        model_types = ["watts_strogatz", "erdos_renyi", "barabasi_albert"]
        results = {}
        
        for model_type in model_types:
            histories = []
            for _ in range(5):  # Run 5 simulations per model
                history = run_single_simulation()
                histories.append(history)
            
            # Calculate metrics
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
        
        # Display Comparison Results
        st.write("### Network Model Comparison")
        
        # Create comparison table
        comparison_data = []
        for model, result in results.items():
            metrics_df = pd.DataFrame(result['metrics'])
            row = {
                'Model': model,
                'Mean R0': metrics_df['R0'].mean(),
                'Mean Peak Infected': metrics_df['peak_infected'].mean(),
                'Mean Total Deaths': metrics_df['total_deaths'].mean(),
                'Mean Epidemic Duration': metrics_df['epidemic_duration'].mean()
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df)
        
        # Plot comparison charts
        st.write("### Comparison Charts")
        
        # R0 Comparison
        fig = go.Figure()
        for model, result in results.items():
            metrics_df = pd.DataFrame(result['metrics'])
            fig.add_trace(go.Box(
                y=metrics_df['R0'],
                name=model,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
        fig.update_layout(
            title="R0 Comparison Across Network Models",
            yaxis_title="R0"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Peak Infected Comparison
        fig = go.Figure()
        for model, result in results.items():
            metrics_df = pd.DataFrame(result['metrics'])
            fig.add_trace(go.Box(
                y=metrics_df['peak_infected'],
                name=model,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
        fig.update_layout(
            title="Peak Infected Comparison Across Network Models",
            yaxis_title="Peak Infected"
        )
        st.plotly_chart(fig, use_container_width=True) 