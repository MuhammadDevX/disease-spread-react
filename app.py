import streamlit as st
from src.generate_network import generate_social_network
from src.visualize_network import visualize_social_network, display_interactive_network
from src.simulate_disease import simulate_disease_spread
from src.simulation_controls import initialize_simulation_state, step_simulation
from src.generate_report import generate_summary
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit Page Config
st.set_page_config(page_title="Interactive Disease Simulation", layout="wide")

st.title("🦠 Interactive Disease Spread Simulation")

# Sidebar — Network & Simulation Parameters
st.sidebar.header("Network & Simulation Settings")

num_nodes = st.sidebar.slider("Number of People", 100, 2000, 500, 50)
model = st.sidebar.selectbox("Network Model", ["watts_strogatz", "erdos_renyi", "barabasi_albert"])
p_infection = st.sidebar.slider("Infection Probability", 0.01, 1.0, 0.08, 0.01)
p_recovery = st.sidebar.slider("Recovery Probability", 0.0, 1.0, 0.05, 0.01)
p_death = st.sidebar.slider("Death Probability", 0.0, 1.0, 0.01, 0.005)
steps_per_run = st.sidebar.slider("Steps per Run", 1, 50, 5, 1)

# Session state for simulation state
if "sim_state" not in st.session_state:
    st.session_state.sim_state = None

# Generate Network Button
if st.sidebar.button("🛠️ Generate New Network"):
    G = generate_social_network(num_nodes, model)
    st.session_state.sim_state = initialize_simulation_state(G)
    st.success("New network generated!")

if st.session_state.sim_state:
    sim_state = st.session_state.sim_state

    st.subheader(f"Interactive Network (Step {sim_state['current_step']})")
    display_interactive_network(sim_state['graph'])

    # Simulation Controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"➡️ Run {steps_per_run} Steps"):
            sim_state = step_simulation(
                sim_state, 
                p_infection, p_recovery, p_death, 
                steps=steps_per_run, 
                simulate_func=simulate_disease_spread
            )
            st.session_state.sim_state = sim_state
            st.experimental_rerun()

    with col2:
        if st.button("🔄 Reset Simulation"):
            G = generate_social_network(num_nodes, model)
            st.session_state.sim_state = initialize_simulation_state(G)
            st.experimental_rerun()

    # Infection Chart
    st.subheader("📈 Population State Progression")
    if sim_state['log']:
        df = pd.DataFrame(sim_state['log'])
        df.index.name = "Step"

        fig, ax = plt.subplots()
        df.plot(ax=ax, marker='o')
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Population Count")
        ax.set_title("Population State Progression Over Time")
        st.pyplot(fig)

    # Generate AI Summary
    st.subheader("📑 AI-Generated Simulation Summary")
    last_counts = sim_state['log'][-1] if sim_state['log'] else {
        'healthy': num_nodes - 5, 'infected': 5, 'recovered': 0, 'dead': 0
    }
    report_text = (
        f"This simulation modeled the spread of a disease in a town of {num_nodes} people. "
        f"The model used was {model} with an infection probability of {p_infection}, "
        f"recovery probability of {p_recovery}, and death probability of {p_death}. "
        f"After {sim_state['current_step']} steps, the final population counts were: "
        f"{last_counts['infected']} infected, {last_counts['recovered']} recovered, "
        f"{last_counts['dead']} dead, and {last_counts['healthy']} remaining healthy."
    )
    with st.spinner("Generating AI Summary..."):
        summary = generate_summary(report_text)
    st.write(summary)

else:
    st.info("⬅️ Generate a network first to start the simulation!")

