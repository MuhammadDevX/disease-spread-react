import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from src.generate_network import generate_social_network
from src.visualize_network import visualize_social_network
from src.simulate_disease import simulate_disease_spread
from src.generate_report import generate_summary

# Streamlit Page Config
st.set_page_config(page_title="Disease Spread Simulation", layout="wide")

st.title("🦠 Disease Spread Simulation in a Social Network")

# Sidebar Parameters
st.sidebar.header("Simulation Settings")

num_nodes = st.sidebar.slider("Number of People", 100, 2000, 500, 50)
model = st.sidebar.selectbox("Network Model", ["watts_strogatz", "erdos_renyi", "barabasi_albert"])
p_infection = st.sidebar.slider("Infection Probability", 0.01, 1.0, 0.08, 0.01)
steps = st.sidebar.slider("Simulation Steps", 1, 50, 10, 1)

# Run Button
if st.sidebar.button("Run Simulation"):
    st.subheader("Generated Social Network")
    G = generate_social_network(num_nodes, model)

    # Visualize Initial State
    fig, ax = plt.subplots(figsize=(8, 6))
    visualize_social_network(G, "Initial State")
    st.pyplot(fig)

    # Simulate
    G = simulate_disease_spread(G, p_infection, steps)

    st.subheader("After Simulation")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    visualize_social_network(G, "After Simulation")
    st.pyplot(fig2)

    # Generate Summary
    report_text = f"This simulation modeled the spread of disease across a town of {num_nodes} people. An initial 5 infections spread over {steps} time steps with a probability of infection set to {p_infection}."
    with st.spinner("Generating Summary using Hugging Face..."):
        summary = generate_summary(report_text)
    st.subheader("📑 AI-Generated Simulation Summary")
    st.write(summary)

else:
    st.info("⬅️ Adjust parameters and click **Run Simulation** to start!")

