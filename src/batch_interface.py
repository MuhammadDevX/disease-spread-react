import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any
from src.batch_simulation import (
    run_batch_simulations,
    analyze_batch_results,
    compare_network_models
)
from src.disease_model import DiseaseParameters
from src.visualize_network import create_time_series_plot

def display_batch_simulation_interface():
    """Display the batch simulation interface in Streamlit."""
    st.header("Batch Simulation Analysis")
    
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
    
    # Run Batch Simulations
    if st.button("Run Batch Simulations"):
        with st.spinner("Running batch simulations..."):
            histories, metrics_list = run_batch_simulations(
                num_simulations,
                num_nodes,
                network_model,
                model_type,
                params,
                num_steps,
                num_workers
            )
            
            analysis = analyze_batch_results(histories, metrics_list)
            
            # Display Results
            st.subheader("Simulation Results")
            
            # Metrics Summary
            st.write("### Key Metrics")
            metrics_df = pd.DataFrame({
                'Mean': analysis['mean_metrics'],
                'Std Dev': analysis['std_metrics'],
                '95% CI Lower': {k: v['lower'] for k, v in analysis.items() if k.endswith('_ci')},
                '95% CI Upper': {k: v['upper'] for k, v in analysis.items() if k.endswith('_ci')}
            })
            st.dataframe(metrics_df)
            
            # Average Time Series
            st.write("### Average Population States Over Time")
            fig = create_time_series_plot(
                analysis['average_history'],
                f"Average {model_type} Model Simulation"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution Plots
            st.write("### Metric Distributions")
            metrics_df = pd.DataFrame(metrics_list)
            
            for metric in ['peak_infected', 'total_deaths', 'R0']:
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
            results = compare_network_models(
                num_nodes,
                model_types,
                params,
                num_steps
            )
            
            # Display Comparison Results
            st.write("### Network Model Comparison")
            
            # Create comparison table
            comparison_data = []
            for model, analysis in results.items():
                row = {
                    'Model': model,
                    'Mean R0': analysis['mean_metrics']['R0'],
                    'Mean Peak Infected': analysis['mean_metrics']['peak_infected'],
                    'Mean Total Deaths': analysis['mean_metrics']['total_deaths'],
                    'Mean Epidemic Duration': analysis['mean_metrics']['epidemic_duration']
                }
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)
            
            # Plot comparison charts
            st.write("### Comparison Charts")
            
            # R0 Comparison
            fig = go.Figure()
            for model, analysis in results.items():
                fig.add_trace(go.Box(
                    y=[m['R0'] for m in metrics_list],
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
            for model, analysis in results.items():
                fig.add_trace(go.Box(
                    y=[m['peak_infected'] for m in metrics_list],
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