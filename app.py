import streamlit as st
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from src.generate_network import generate_social_network
from src.disease_model import DiseaseModel, DiseaseParameters, DiseaseState
from src.policy_engine import PolicyEngine, Policy, PolicyTrigger
from src.visualize_network import create_network_figure, create_time_series_plot, create_heatmap, create_animated_infection_visualization
from src.batch_interface import display_batch_simulation_interface
import pandas as pd
from typing import Dict, List
import time

# Page Config
st.set_page_config(
    page_title="Advanced Disease Spread Simulation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🦠 Advanced Disease Spread Simulation")
st.markdown("""
This simulation models disease spread in social networks with various intervention strategies.
Explore how different parameters and policies affect the spread of disease.
""")

# Sidebar
st.sidebar.header("Simulation Parameters")

# Network Parameters
st.sidebar.subheader("Network Settings")
num_nodes = st.sidebar.slider("Population Size", 100, 2000, 500, 50)
network_model = st.sidebar.selectbox(
    "Network Model",
    ["watts_strogatz", "erdos_renyi", "barabasi_albert"],
    help="Different network topologies simulate different social structures"
)

# Disease Model Parameters
st.sidebar.subheader("Disease Model")
model_type = st.sidebar.selectbox(
    "Model Type",
    ["SIR", "SEIR", "SIRD"],
    help="SIR: Basic model, SEIR: Includes exposed state, SIRD: Includes death state"
)

transmission_rate = st.sidebar.slider(
    "Transmission Rate (β)",
    0.0, 1.0, 0.5, 0.01,
    help="Probability of disease transmission per contact"
)
recovery_rate = st.sidebar.slider(
    "Recovery Rate (γ)",
    0.0, 1.0, 0.1, 0.01,
    help="Probability of recovery per timestep"
)
mortality_rate = st.sidebar.slider(
    "Mortality Rate (δ)",
    0.0, 1.0, 0.01, 0.005,
    help="Probability of death instead of recovery"
)

# Healthcare Parameters
st.sidebar.subheader("Healthcare System")
healthcare_capacity = st.sidebar.slider(
    "Healthcare Capacity",
    10, 500, 100, 10,
    help="Maximum number of simultaneous infections before healthcare system overwhelmed"
)

# Initial Infection Parameters
st.sidebar.subheader("Initial Conditions")
initial_infection_rate = st.sidebar.slider(
    "Initial Infection Rate",
    0.0, 0.1, 0.01, 0.001,
    help="Percentage of population initially infected"
)

# Policy Parameters
st.sidebar.subheader("Intervention Policies")
lockdown_threshold = st.sidebar.slider(
    "Lockdown Threshold",
    0.0, 1.0, 0.2, 0.01,
    help="Infection rate at which lockdown measures are implemented"
)
vaccination_coverage = st.sidebar.slider(
    "Vaccination Coverage",
    0.0, 1.0, 0.0, 0.05,
    help="Percentage of population to vaccinate"
)
vaccination_strategy = st.sidebar.selectbox(
    "Vaccination Strategy",
    ["random", "targeted"],
    help="Random: Uniform distribution, Targeted: Focus on high-degree nodes"
)

# Initialize session state
if "simulation" not in st.session_state:
    st.session_state.simulation = None
if "policy_engine" not in st.session_state:
    st.session_state.policy_engine = PolicyEngine()
if "history" not in st.session_state:
    st.session_state.history = []

# Generate Network Button
if st.sidebar.button("🛠️ Generate New Network"):
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
    
    # Add policies
    st.session_state.policy_engine.add_policy(Policy(
        name="Lockdown",
        trigger=PolicyTrigger.INFECTION_THRESHOLD,
        threshold=lockdown_threshold,
        action=lambda: st.session_state.simulation.apply_lockdown(lockdown_threshold),
        params={}
    ))
    
    if vaccination_coverage > 0:
        st.session_state.policy_engine.add_policy(Policy(
            name="Vaccination",
            trigger=PolicyTrigger.INFECTION_THRESHOLD,
            threshold=0.0,  # Immediate vaccination
            action=lambda: st.session_state.simulation.apply_vaccination(
                vaccination_coverage,
                vaccination_strategy
            ),
            params={}
        ))
    
    st.success("New network generated with specified parameters!")

# Main Simulation Area
if st.session_state.simulation:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Network Visualization")
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Static View", "Animated Infection Spread"])
        
        with tab1:
            # Static network visualization
            fig = create_network_figure(
                st.session_state.simulation.graph,
                st.session_state.simulation.states,
                st.session_state.history[-1].get("infection_sources", {}) if st.session_state.history else {}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Animated infection visualization
            if st.session_state.history and "infection_sources" in st.session_state.history[-1]:
                anim_fig = create_animated_infection_visualization(
                    st.session_state.simulation.graph,
                    st.session_state.simulation.states,
                    st.session_state.history[-1]["infection_sources"]
                )
                st.plotly_chart(anim_fig, use_container_width=True)
            else:
                st.info("No new infections to animate.")
        
        # Display infection history
        if st.session_state.history and "infection_sources" in st.session_state.history[-1]:
            st.subheader("Recent Infections")
            infection_sources = st.session_state.history[-1]["infection_sources"]
            if infection_sources:
                infection_data = []
                for infected, source in infection_sources.items():
                    infection_data.append({
                        "Infected Node": infected,
                        "Source Node": source,
                        "Time Step": len(st.session_state.history) - 1
                    })
                st.dataframe(pd.DataFrame(infection_data))
            else:
                st.info("No new infections in the last step.")
        
        # Create heatmap
        st.subheader("Infection Density Heatmap")
        heatmap_fig = create_heatmap(
            st.session_state.simulation.graph,
            st.session_state.simulation.states
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)

    with col2:
        st.subheader("Simulation Controls")
        if st.button("▶️ Run One Step"):
            state_counts = st.session_state.simulation.step()
            st.session_state.history.append(state_counts)
            st.session_state.policy_engine.evaluate_policies(state_counts)
            st.rerun()
            
        if st.button("⏩ Run 10 Steps"):
            for _ in range(10):
                state_counts = st.session_state.simulation.step()
                st.session_state.history.append(state_counts)
                st.session_state.policy_engine.evaluate_policies(state_counts)
            st.rerun()
            
        if st.button("🔄 Reset Simulation"):
            st.session_state.simulation = None
            st.session_state.policy_engine = PolicyEngine()
            st.session_state.history = []
            st.rerun()

    # Time Series Plot
    st.subheader("Population State Over Time")
    if st.session_state.history:
        fig = create_time_series_plot(
            st.session_state.history,
            f"{model_type} Model Simulation"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Policy History
    st.subheader("Policy Interventions")
    policy_history = st.session_state.policy_engine.get_policy_history()
    if policy_history:
        st.dataframe(pd.DataFrame(policy_history))
    else:
        st.info("No policies have been activated yet.")
    
    # Current Statistics
    st.subheader("Current Statistics")
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
            
        # Healthcare Status
        if st.session_state.simulation.healthcare_overwhelmed:
            st.error("⚠️ Healthcare System Overwhelmed")
        else:
            st.success("✅ Healthcare System Operating Normally")

else:
    st.info("⬅️ Configure parameters and generate a network to start the simulation!")

# Batch Simulation Interface
st.markdown("---")
display_batch_simulation_interface()

