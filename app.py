import streamlit as st
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from src.generate_network import generate_social_network
from src.disease_model import DiseaseModel, DiseaseParameters, DiseaseState
from src.policy_engine import PolicyEngine, Policy, PolicyTrigger
from src.visualize_network import create_network_figure, create_time_series_plot, create_heatmap, create_animated_infection_visualization
import pandas as pd
from typing import Dict, List
import time

# Page Config must be the first Streamlit command
st.set_page_config(
    page_title="Disease Spread Simulation",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
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
            padding: 1rem;
        display: flex;
        gap: 1.5rem;
        font-size: 1rem;
        justify-content: center;

    }
    
    .navbar-link {
            color: #ADD8E6  !important;
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

    .welcome-card{
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
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

# Main content
st.markdown("""
<div class="welcome-card">
    <h2>Welcome to the Disease Spread Simulation</h2>
    <p>This simulation models disease spread in social networks with various intervention strategies.
    Explore how different parameters and policies affect the spread of disease.</p>
</div>
""", unsafe_allow_html=True)

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
            help="Different network topologies simulate different social structures"
        )
    
    with col2:
        model_type = st.selectbox(
            "Model Type",
            ["SIR", "SEIR", "SIRD"],
            help="SIR: Basic model, SEIR: Includes exposed state, SIRD: Includes death state"
        )

# Disease Parameters
with st.expander("Disease Parameters", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        transmission_rate = st.slider(
            "Transmission Rate (β)",
            0.0, 1.0, 0.5, 0.01,
            help="Probability of disease transmission per contact"
        )
    
    with col2:
        recovery_rate = st.slider(
            "Recovery Rate (γ)",
            0.0, 1.0, 0.1, 0.01,
            help="Probability of recovery per timestep"
        )
    
    with col3:
        mortality_rate = st.slider(
            "Mortality Rate (δ)",
            0.0, 1.0, 0.01, 0.005,
            help="Probability of death instead of recovery"
        )

# Healthcare Parameters
with st.expander("Healthcare System", expanded=True):
    healthcare_capacity = st.slider(
        "Healthcare Capacity",
        10, 500, 100, 10,
        help="Maximum number of simultaneous infections before healthcare system overwhelmed"
    )

# Initial Infection Parameters
with st.expander("Initial Conditions", expanded=True):
    initial_infection_rate = st.slider(
        "Initial Infection Rate",
        0.0, 0.1, 0.01, 0.001,
        help="Percentage of population initially infected"
    )

# Initialize session state
if "simulation" not in st.session_state:
    st.session_state.simulation = None
if "policy_engine" not in st.session_state:
    st.session_state.policy_engine = PolicyEngine()
if "history" not in st.session_state:
    st.session_state.history = []

# Generate Network Button
if st.button("🛠️ Generate New Network", key="generate_network"):
    G = generate_social_network(num_nodes, network_model)
    params = DiseaseParameters(
        transmission_rate=transmission_rate,
        recovery_rate=recovery_rate,
        mortality_rate=mortality_rate,
        latency_period=5,
        immunity_duration=100,
        healthcare_capacity=healthcare_capacity
    )
    st.session_state.simulation = DiseaseModel(G, model_type, params)
    
    # Seed initial infections
    num_initial_infections = int(num_nodes * initial_infection_rate)
    if num_initial_infections > 0:
        nodes_to_infect = np.random.choice(
            list(G.nodes()),
            size=num_initial_infections,
            replace=False
        )
        for node in nodes_to_infect:
            st.session_state.simulation.states[node] = DiseaseState.INFECTED
            st.session_state.simulation.time_in_state[node] = 0
    
    st.session_state.policy_engine = PolicyEngine()
    st.session_state.history = []
    st.success("New network generated with specified parameters!")

# Main Simulation Area
if st.session_state.simulation:
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Network View", "Animated Spread", "Statistics", "Heatmap"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Network Visualization")
            fig = create_network_figure(
                st.session_state.simulation.graph,
                st.session_state.simulation.states,
                st.session_state.history[-1].get("infection_sources", {}) if st.session_state.history else {}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Simulation Controls")
            if st.button("▶️ Run One Step", key="step"):
                state_counts = st.session_state.simulation.step()
                st.session_state.history.append(state_counts)
                st.session_state.policy_engine.evaluate_policies(state_counts)
                st.rerun()
            
            if st.button("⏩ Run 10 Steps", key="run_10"):
                for _ in range(10):
                    state_counts = st.session_state.simulation.step()
                    st.session_state.history.append(state_counts)
                    st.session_state.policy_engine.evaluate_policies(state_counts)
                st.rerun()
            
            if st.button("🔄 Reset Simulation", key="reset"):
                st.session_state.simulation = None
                st.session_state.policy_engine = PolicyEngine()
                st.session_state.history = []
                st.rerun()
    
    with tab2:
        st.markdown("### Animated Disease Spread")
        if st.session_state.history and "infection_sources" in st.session_state.history[-1]:
            anim_fig = create_animated_infection_visualization(
                st.session_state.simulation.graph,
                st.session_state.simulation.states,
                st.session_state.history[-1]["infection_sources"]
            )
            st.plotly_chart(anim_fig, use_container_width=True)
        else:
            st.info("No new infections to animate.")
    
    with tab3:
        # Time Series Plot
        st.markdown("### Population State Over Time")
        if st.session_state.history:
            fig = create_time_series_plot(
                st.session_state.history,
                f"{model_type} Model Simulation"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Current Statistics
        st.markdown("### Current Statistics")
        if st.session_state.history:
            current_state = st.session_state.history[-1]
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Susceptible", current_state.get("S", 0))
            with col2:
                st.metric("Exposed", current_state.get("E", 0))
            with col3:
                st.metric("Infected", current_state.get("I", 0))
            with col4:
                st.metric("Recovered", current_state.get("R", 0))
            with col5:
                st.metric("Deceased", current_state.get("D", 0))
    
    with tab4:
        st.markdown("### Infection Density Heatmap")
        heatmap_fig = create_heatmap(
            st.session_state.simulation.graph,
            st.session_state.simulation.states
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)

else:
    st.info("⬅️ Configure parameters and generate a network to start the simulation!")

