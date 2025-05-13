import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import networkx as nx
import pandas as pd
import pickle
from src.disease_model import DiseaseModel, DiseaseParameters, DiseaseState
from src.generate_network import generate_social_network
from src.visualize_network import create_network_figure, create_time_series_plot, create_heatmap, create_animated_infection_visualization
from src.policy_engine import PolicyEngine, Policy, PolicyTrigger

# App initialization
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Disease Spread Simulation'
# Allow callbacks for components not in the initial layout (required for multi-page apps)
app.config.suppress_callback_exceptions = True

# Navigation bar
navbar = dbc.Navbar(
    dbc.Container([
        html.A(
            dbc.Row([
                dbc.Col(html.Div("🦠 Disease Spread Simulation", className="navbar-title", style={"fontWeight": "bold", "fontSize": "1.5rem"})),
            ], align="center", className="g-0"),
            href="/",
            style={"textDecoration": "none"},
        ),
        dbc.Nav([
            dbc.NavLink("Main Simulation", href="/", active="exact", className="navbar-link"),
            dbc.NavLink("Batch Analysis", href="/batch_simulation", active="exact", className="navbar-link"),
            dbc.NavLink("Policy Interventions", href="/policy_interventions", active="exact", className="navbar-link"),
            dbc.NavLink("Model Comparison", href="/model_comparison", active="exact", className="navbar-link"),
        ], className="navbar-links", pills=True),
    ]),
    color="light",
    dark=False,
    className="mb-4 navbar"
)

# App layout with location for routing
app.layout = html.Div([
    dcc.Location(id="url"),
    navbar,
    # All dcc.Store and callback output divs for all pages (must be present at all times):
    dcc.Store(id="sim-session", storage_type="session"),
    dcc.Store(id="batch-session", storage_type="session"),
    dcc.Store(id="policy-session", storage_type="session"),
    dcc.Store(id="modelcomp-session", storage_type="session"),
    html.Div(id="sim-message", style={"display": "none"}),
    html.Div(id="batch-message", style={"display": "none"}),
    html.Div(id="policy-message", style={"display": "none"}),
    html.Div(id="modelcomp-message", style={"display": "none"}),
    html.Div(id="sim-tab-content", style={"display": "none"}),
    html.Div(id="batch-results", style={"display": "none"}),
    html.Div(id="policy-results", style={"display": "none"}),
    html.Div(id="modelcomp-results", style={"display": "none"}),
    html.Div(id="page-content")
])

# Main simulation layout

def main_simulation_layout():
    return html.Div([
        html.H2("Disease Spread Simulation (Dash Version)"),
        html.P("This simulation models disease spread in social networks with various intervention strategies."),
        dbc.Row([
            dbc.Col([
                html.H4("Network Settings"),
                dbc.Label("Population Size"),
                dcc.Slider(id="num-nodes", min=100, max=2000, step=10, value=500, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Network Model", className="mt-2"),
                dcc.Dropdown(id="network-model", options=[
                    {"label": "Watts-Strogatz", "value": "watts_strogatz"},
                    {"label": "Erdős-Rényi", "value": "erdos_renyi"},
                    {"label": "Barabási-Albert", "value": "barabasi_albert"},
                ], value="watts_strogatz"),
                dbc.Label("Model Type", className="mt-2"),
                dcc.Dropdown(id="model-type", options=[
                    {"label": "SIR", "value": "SIR"},
                    {"label": "SEIR", "value": "SEIR"},
                    {"label": "SIRD", "value": "SIRD"},
                ], value="SIR"),
                html.Hr(),
                html.H4("Disease Parameters"),
                dbc.Label("Transmission Rate (β)"),
                dcc.Slider(id="transmission-rate", min=0, max=1, step=0.01, value=0.5, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Recovery Rate (γ)", className="mt-2"),
                dcc.Slider(id="recovery-rate", min=0, max=1, step=0.01, value=0.1, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Mortality Rate (δ)", className="mt-2"),
                dcc.Slider(id="mortality-rate", min=0, max=1, step=0.005, value=0.01, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                html.Hr(),
                html.H4("Healthcare System"),
                dbc.Label("Healthcare Capacity"),
                dcc.Slider(id="healthcare-capacity", min=10, max=500, step=10, value=100, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                html.Hr(),
                html.H4("Initial Conditions"),
                dbc.Label("Initial Infection Rate"),
                dcc.Slider(id="initial-infection-rate", min=0, max=0.1, step=0.001, value=0.01, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                html.Br(),
                dbc.Button("Generate New Network", id="generate-network", color="primary", className="mb-2", n_clicks=0),
                dbc.Button("Run One Step", id="run-step", color="success", className="mb-2 ms-2", n_clicks=0),
                dbc.Button("Run 10 Steps", id="run-10", color="info", className="mb-2 ms-2", n_clicks=0),
                dbc.Button("Reset Simulation", id="reset-sim", color="danger", className="mb-2 ms-2", n_clicks=0),
            ], width=3),
            dbc.Col([
                dcc.Tabs(id="sim-tabs", value="network", children=[
                    dcc.Tab(label="Network View", value="network"),
                    dcc.Tab(label="Animated Spread", value="animated"),
                    dcc.Tab(label="Statistics", value="stats"),
                    dcc.Tab(label="Heatmap", value="heatmap"),
                ]),
                html.Div(id="sim-tab-content", className="mt-3"),
            ], width=9)
        ])
    ], className="container")

def batch_simulation_layout():
    return html.Div([
        html.H2("Batch Simulation Analysis"),
        html.P("Run multiple simulations in parallel to analyze disease spread patterns across different network models and parameters."),
        dbc.Row([
            dbc.Col([
                html.H4("Batch Parameters"),
                dbc.Label("Number of Simulations"),
                dcc.Slider(id="batch-num-sim", min=1, max=100, step=1, value=10, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Steps per Simulation", className="mt-2"),
                dcc.Slider(id="batch-num-steps", min=10, max=500, step=10, value=100, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Number of Workers", className="mt-2"),
                dcc.Slider(id="batch-num-workers", min=1, max=8, step=1, value=4, tooltip={"placement": "bottom", "always_visible": True}),
                html.Hr(),
                html.H4("Network & Disease Parameters"),
                dbc.Label("Population Size"),
                dcc.Slider(id="batch-num-nodes", min=100, max=2000, step=10, value=500, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Network Model", className="mt-2"),
                dcc.Dropdown(id="batch-network-model", options=[
                    {"label": "Watts-Strogatz", "value": "watts_strogatz"},
                    {"label": "Erdős-Rényi", "value": "erdos_renyi"},
                    {"label": "Barabási-Albert", "value": "barabasi_albert"},
                ], value="watts_strogatz"),
                dbc.Label("Disease Model", className="mt-2"),
                dcc.Dropdown(id="batch-model-type", options=[
                    {"label": "SIR", "value": "SIR"},
                    {"label": "SEIR", "value": "SEIR"},
                    {"label": "SIRD", "value": "SIRD"},
                ], value="SIR"),
                html.Hr(),
                dbc.Label("Transmission Rate (β)"),
                dcc.Slider(id="batch-transmission-rate", min=0, max=1, step=0.01, value=0.3, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Recovery Rate (γ)", className="mt-2"),
                dcc.Slider(id="batch-recovery-rate", min=0, max=1, step=0.01, value=0.1, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Mortality Rate (δ)", className="mt-2"),
                dcc.Slider(id="batch-mortality-rate", min=0, max=1, step=0.005, value=0.01, tooltip={"placement": "bottom", "always_visible": True}),
                html.Br(),
                dbc.Button("Run Batch Simulations", id="run-batch", color="primary", className="mb-2", n_clicks=0),
            ], width=3),
            dbc.Col([
                html.Div(id="batch-results", className="mt-3"),
            ], width=9)
        ])
    ], className="container")

def policy_interventions_layout():
    return html.Div([
        html.H2("Policy Interventions Analysis"),
        html.P("Analyze the effectiveness of different intervention policies on disease spread, including lockdowns, vaccinations, and healthcare capacity management."),
        dbc.Row([
            dbc.Col([
                html.H4("Network and Disease Parameters"),
                dbc.Label("Population Size"),
                dcc.Slider(id="policy-num-nodes", min=100, max=2000, step=10, value=500, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Network Model", className="mt-2"),
                dcc.Dropdown(id="policy-network-model", options=[
                    {"label": "Watts-Strogatz", "value": "watts_strogatz"},
                    {"label": "Erdős-Rényi", "value": "erdos_renyi"},
                    {"label": "Barabási-Albert", "value": "barabasi_albert"},
                ], value="watts_strogatz"),
                html.Hr(),
                dbc.Label("Transmission Rate (β)"),
                dcc.Slider(id="policy-transmission-rate", min=0, max=1, step=0.01, value=0.3, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Recovery Rate (γ)", className="mt-2"),
                dcc.Slider(id="policy-recovery-rate", min=0, max=1, step=0.01, value=0.1, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Mortality Rate (δ)", className="mt-2"),
                dcc.Slider(id="policy-mortality-rate", min=0, max=1, step=0.005, value=0.01, tooltip={"placement": "bottom", "always_visible": True}),
                html.Hr(),
                html.H4("Intervention Settings"),
                dbc.Label("Lockdown: Infection Rate Threshold"),
                dcc.Slider(id="lockdown-threshold", min=0, max=1, step=0.01, value=0.2, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Lockdown Strength", className="mt-2"),
                dcc.Slider(id="lockdown-strength", min=0, max=1, step=0.05, value=0.3, tooltip={"placement": "bottom", "always_visible": True}),
                html.Hr(),
                dbc.Label("Vaccination Coverage"),
                dcc.Slider(id="vaccination-coverage", min=0, max=1, step=0.05, value=0.0, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Vaccination Strategy", className="mt-2"),
                dcc.Dropdown(id="vaccination-strategy", options=[
                    {"label": "Random", "value": "random"},
                    {"label": "Targeted", "value": "targeted"},
                ], value="random"),
                html.Hr(),
                dbc.Label("Healthcare Capacity"),
                dcc.Slider(id="policy-healthcare-capacity", min=10, max=500, step=10, value=100, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Healthcare Effectiveness", className="mt-2"),
                dcc.Slider(id="healthcare-effectiveness", min=0, max=1, step=0.1, value=0.5, tooltip={"placement": "bottom", "always_visible": True}),
                html.Br(),
                dbc.Button("Run Policy Analysis", id="run-policy-analysis", color="primary", className="mb-2", n_clicks=0),
            ], width=3),
            dbc.Col([
                html.Div(id="policy-results", className="mt-3"),
            ], width=9)
        ])
    ], className="container")

def model_comparison_layout():
    return html.Div([
        html.H2("Model Comparison"),
        html.P("Compare different disease spread models (SIR, SEIR, SIRD) and network models to understand their impact on disease spread patterns."),
        dbc.Row([
            dbc.Col([
                html.H4("Network Settings"),
                dbc.Label("Population Size"),
                dcc.Slider(id="mc-num-nodes", min=100, max=2000, step=10, value=500, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Network Model", className="mt-2"),
                dcc.Dropdown(id="mc-network-model", options=[
                    {"label": "Watts-Strogatz", "value": "watts_strogatz"},
                    {"label": "Erdős-Rényi", "value": "erdos_renyi"},
                    {"label": "Barabási-Albert", "value": "barabasi_albert"},
                ], value="watts_strogatz"),
                dbc.Label("Select Models to Compare", className="mt-2"),
                dcc.Dropdown(id="mc-selected-models", options=[
                    {"label": "SIR", "value": "SIR"},
                    {"label": "SEIR", "value": "SEIR"},
                    {"label": "SIRD", "value": "SIRD"},
                ], value=["SIR", "SEIR"], multi=True),
                html.Hr(),
                html.H4("Disease Parameters"),
                dbc.Label("Transmission Rate (β)"),
                dcc.Slider(id="mc-transmission-rate", min=0, max=1, step=0.01, value=0.3, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Recovery Rate (γ)", className="mt-2"),
                dcc.Slider(id="mc-recovery-rate", min=0, max=1, step=0.01, value=0.1, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Mortality Rate (δ)", className="mt-2"),
                dcc.Slider(id="mc-mortality-rate", min=0, max=1, step=0.005, value=0.01, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Latency Period (λ)", className="mt-2"),
                dcc.Slider(id="mc-latency-period", min=1, max=14, step=1, value=5, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Immunity Duration", className="mt-2"),
                dcc.Slider(id="mc-immunity-duration", min=10, max=365, step=1, value=100, tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Healthcare Capacity", className="mt-2"),
                dcc.Slider(id="mc-healthcare-capacity", min=10, max=500, step=10, value=100, tooltip={"placement": "bottom", "always_visible": True}),
                html.Br(),
                dbc.Button("Run Model Comparison", id="run-modelcomp", color="primary", className="mb-2", n_clicks=0),
            ], width=3),
            dbc.Col([
                html.Div(id="modelcomp-results", className="mt-3"),
            ], width=9)
        ])
    ], className="container")

# Routing callback
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/batch_simulation":
        return batch_simulation_layout()
    elif pathname == "/policy_interventions":
        return policy_interventions_layout()
    elif pathname == "/model_comparison":
        return model_comparison_layout()
    else:
        return main_simulation_layout()

# --- Simulation State Management ---

def get_default_simulation():
    return dict(sim=None, history=[], policy_engine=None)

@app.callback(
    Output("sim-session", "data"),
    Output("sim-message", "children"),
    Input("generate-network", "n_clicks"),
    Input("run-step", "n_clicks"),
    Input("run-10", "n_clicks"),
    Input("reset-sim", "n_clicks"),
    State("num-nodes", "value"),
    State("network-model", "value"),
    State("model-type", "value"),
    State("transmission-rate", "value"),
    State("recovery-rate", "value"),
    State("mortality-rate", "value"),
    State("healthcare-capacity", "value"),
    State("initial-infection-rate", "value"),
    State("sim-session", "data"),
    prevent_initial_call=True
)
def update_simulation(generate_clicks, step_clicks, run10_clicks, reset_clicks,
                      num_nodes, network_model, model_type, transmission_rate, recovery_rate, mortality_rate,
                      healthcare_capacity, initial_infection_rate, sim_data):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    msg = None
    if sim_data is None:
        sim_data = get_default_simulation()
    # Generate new network
    if button_id == "generate-network":
        G = generate_social_network(num_nodes, network_model)
        params = DiseaseParameters(
            transmission_rate=transmission_rate,
            recovery_rate=recovery_rate,
            mortality_rate=mortality_rate,
            latency_period=5,
            immunity_duration=100,
            healthcare_capacity=healthcare_capacity
        )
        sim = DiseaseModel(G, model_type, params)
        # Seed initial infections
        num_initial_infections = int(num_nodes * initial_infection_rate)
        if num_initial_infections > 0:
            nodes_to_infect = np.random.choice(list(G.nodes()), size=num_initial_infections, replace=False)
            for node in nodes_to_infect:
                sim.states[node] = DiseaseState.INFECTED
                sim.time_in_state[node] = 0
        sim_data = dict(sim=pickle.dumps(sim), history=[], policy_engine=None)
        msg = dbc.Alert("New network generated with specified parameters!", color="success")
    elif button_id == "run-step":
        if sim_data.get("sim") is None:
            msg = dbc.Alert("Please generate a network first!", color="warning")
        else:
            sim = pickle.loads(sim_data["sim"])
            history = sim_data.get("history", [])
            state_counts = sim.step()
            history.append(state_counts)
            sim_data = dict(sim=pickle.dumps(sim), history=history, policy_engine=None)
    elif button_id == "run-10":
        if sim_data.get("sim") is None:
            msg = dbc.Alert("Please generate a network first!", color="warning")
        else:
            sim = pickle.loads(sim_data["sim"])
            history = sim_data.get("history", [])
            for _ in range(10):
                state_counts = sim.step()
                history.append(state_counts)
            sim_data = dict(sim=pickle.dumps(sim), history=history, policy_engine=None)
    elif button_id == "reset-sim":
        sim_data = get_default_simulation()
        msg = dbc.Alert("Simulation reset.", color="info")
    return sim_data, msg

# --- Simulation Tab Content ---
@app.callback(
    Output("sim-tab-content", "children"),
    Input("sim-tabs", "value"),
    State("sim-session", "data")
)
def render_simulation_tab(tab, sim_data):
    if sim_data is None or sim_data.get("sim") is None:
        return dbc.Alert("Configure parameters and generate a network to start the simulation!", color="info")
    sim = pickle.loads(sim_data["sim"])
    history = sim_data.get("history", [])
    if tab == "network":
        if not history:
            return dbc.Alert("Run the simulation to see the network visualization.", color="info")
        last_state = history[-1]
        fig = create_network_figure(sim.graph, sim.states, last_state.get("infection_sources", {}))
        return dcc.Graph(figure=fig, config={"displayModeBar": True, "responsive": True})
    elif tab == "animated":
        if not history or "infection_sources" not in history[-1]:
            return dbc.Alert("No new infections to animate.", color="info")
        anim_fig = create_animated_infection_visualization(sim.graph, sim.states, history[-1]["infection_sources"])
        return dcc.Graph(figure=anim_fig, config={"displayModeBar": True, "responsive": True})
    elif tab == "stats":
        if not history:
            return dbc.Alert("Run the simulation to see statistics.", color="info")
        fig = create_time_series_plot(history, f"{sim.model_type} Model Simulation")
        # Current statistics
        current_state = history[-1]
        metrics = [
            dbc.Col(dbc.Card([dbc.CardHeader("Susceptible"), dbc.CardBody(str(current_state.get("S", 0)))])),
            dbc.Col(dbc.Card([dbc.CardHeader("Exposed"), dbc.CardBody(str(current_state.get("E", 0)))])),
            dbc.Col(dbc.Card([dbc.CardHeader("Infected"), dbc.CardBody(str(current_state.get("I", 0)))])),
            dbc.Col(dbc.Card([dbc.CardHeader("Recovered"), dbc.CardBody(str(current_state.get("R", 0)))])),
            dbc.Col(dbc.Card([dbc.CardHeader("Deceased"), dbc.CardBody(str(current_state.get("D", 0)))])),
        ]
        return html.Div([
            dcc.Graph(figure=fig, config={"displayModeBar": True, "responsive": True}),
            html.Hr(),
            dbc.Row(metrics)
        ])
    elif tab == "heatmap":
        if not history:
            return dbc.Alert("Run the simulation to see the heatmap.", color="info")
        heatmap_fig = create_heatmap(sim.graph, sim.states)
        return dcc.Graph(figure=heatmap_fig, config={"displayModeBar": True, "responsive": True})
    return None

def run_single_batch_simulation(num_nodes, network_model, model_type, params, num_steps):
    G = generate_social_network(num_nodes, network_model)
    model = DiseaseModel(G, model_type, params)
    # Seed initial infections (5% of population)
    num_initial_infections = int(num_nodes * 0.05)
    nodes_to_infect = np.random.choice(list(G.nodes()), size=num_initial_infections, replace=False)
    for node in nodes_to_infect:
        model.states[node] = DiseaseState.INFECTED
        model.time_in_state[node] = 0
    history = []
    for _ in range(num_steps):
        state_counts = model.step()
        history.append(state_counts)
    return history

@app.callback(
    Output("batch-session", "data"),
    Output("batch-message", "children"),
    Input("run-batch", "n_clicks"),
    State("batch-num-sim", "value"),
    State("batch-num-steps", "value"),
    State("batch-num-nodes", "value"),
    State("batch-network-model", "value"),
    State("batch-model-type", "value"),
    State("batch-transmission-rate", "value"),
    State("batch-recovery-rate", "value"),
    State("batch-mortality-rate", "value"),
    prevent_initial_call=True
)
def run_batch_simulations(n_clicks, num_sim, num_steps, num_nodes, network_model, model_type, transmission_rate, recovery_rate, mortality_rate):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate
    params = DiseaseParameters(
        transmission_rate=transmission_rate,
        recovery_rate=recovery_rate,
        mortality_rate=mortality_rate,
        latency_period=5,
        immunity_duration=100,
        healthcare_capacity=100
    )
    histories = []
    for _ in range(num_sim):
        history = run_single_batch_simulation(num_nodes, network_model, model_type, params, num_steps)
        histories.append(history)
    msg = dbc.Alert(f"Batch of {num_sim} simulations completed!", color="success")
    return {"histories": pickle.dumps(histories), "num_steps": num_steps, "model_type": model_type}, msg

@app.callback(
    Output("batch-results", "children"),
    Input("batch-session", "data")
)
def display_batch_results(batch_data):
    if not batch_data or "histories" not in batch_data:
        return None
    histories = pickle.loads(batch_data["histories"])
    num_steps = batch_data["num_steps"]
    model_type = batch_data["model_type"]
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
    metrics_df = pd.DataFrame(metrics_list)
    summary_df = pd.DataFrame({
        'Mean': metrics_df.mean(),
        'Std Dev': metrics_df.std(),
        'Min': metrics_df.min(),
        'Max': metrics_df.max()
    })
    # Average time series
    avg_history = []
    for step in range(num_steps):
        step_counts = {}
        for state in ['S', 'E', 'I', 'R', 'D']:
            counts = [history[step].get(state, 0) for history in histories if step < len(history)]
            step_counts[state] = np.mean(counts) if counts else 0
        avg_history.append(step_counts)
    avg_fig = create_time_series_plot(avg_history, f"Average {model_type} Model Simulation")
    # Distribution plots
    dist_figs = []
    for metric in ['peak_infected', 'total_deaths', 'attack_rate']:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=metrics_df[metric], name=metric, nbinsx=20))
        fig.update_layout(title=f"Distribution of {metric}", xaxis_title=metric, yaxis_title="Count")
        dist_figs.append(dcc.Graph(figure=fig, config={"displayModeBar": True, "responsive": True}))
    return html.Div([
        html.H4("Key Metrics Summary"),
        dbc.Table.from_dataframe(summary_df.round(3).reset_index(), striped=True, bordered=True, hover=True),
        html.Hr(),
        html.H4("Average Population States Over Time"),
        dcc.Graph(figure=avg_fig, config={"displayModeBar": True, "responsive": True}),
        html.Hr(),
        html.H4("Metric Distributions"),
        *dist_figs
    ])

# --- Policy Interventions Callbacks ---
def run_policy_simulation(num_nodes, network_model, transmission_rate, recovery_rate, mortality_rate, healthcare_capacity, lockdown_threshold, lockdown_strength, vaccination_coverage, vaccination_strategy, with_policies=True):
    G = generate_social_network(num_nodes, network_model)
    params = DiseaseParameters(
        transmission_rate=transmission_rate,
        recovery_rate=recovery_rate,
        mortality_rate=mortality_rate,
        latency_period=5,
        immunity_duration=100,
        healthcare_capacity=healthcare_capacity
    )
    model = DiseaseModel(G, "SIR", params)
    # Seed initial infections (5% of population)
    num_initial_infections = int(num_nodes * 0.05)
    nodes_to_infect = np.random.choice(list(G.nodes()), size=num_initial_infections, replace=False)
    for node in nodes_to_infect:
        model.states[node] = DiseaseState.INFECTED
        model.time_in_state[node] = 0
    policy_engine = None
    policy_activations = []
    if with_policies:
        policy_engine = PolicyEngine()
        # Lockdown policy
        policy_engine.add_policy(Policy(
            name="Lockdown",
            trigger=PolicyTrigger.INFECTION_THRESHOLD,
            threshold=lockdown_threshold,
            action=lambda: model.apply_lockdown(lockdown_strength),
            params={}
        ))
        # Vaccination policy
        if vaccination_coverage > 0:
            policy_engine.add_policy(Policy(
                name="Vaccination",
                trigger=PolicyTrigger.INFECTION_THRESHOLD,
                threshold=0.0,
                action=lambda: model.apply_vaccination(vaccination_coverage, vaccination_strategy),
                params={}
            ))
    history = []
    for step in range(100):
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

@app.callback(
    Output("policy-session", "data"),
    Output("policy-message", "children"),
    Input("run-policy-analysis", "n_clicks"),
    State("policy-num-nodes", "value"),
    State("policy-network-model", "value"),
    State("policy-transmission-rate", "value"),
    State("policy-recovery-rate", "value"),
    State("policy-mortality-rate", "value"),
    State("policy-healthcare-capacity", "value"),
    State("lockdown-threshold", "value"),
    State("lockdown-strength", "value"),
    State("vaccination-coverage", "value"),
    State("vaccination-strategy", "value"),
    State("healthcare-effectiveness", "value"),
    prevent_initial_call=True
)
def run_policy_analysis(n_clicks, num_nodes, network_model, transmission_rate, recovery_rate, mortality_rate, healthcare_capacity, lockdown_threshold, lockdown_strength, vaccination_coverage, vaccination_strategy, healthcare_effectiveness):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate
    # With policies
    history_with_policies, policy_activations = run_policy_simulation(
        num_nodes, network_model, transmission_rate, recovery_rate, mortality_rate, healthcare_capacity,
        lockdown_threshold, lockdown_strength, vaccination_coverage, vaccination_strategy, with_policies=True
    )
    # Without policies
    history_without_policies, _ = run_policy_simulation(
        num_nodes, network_model, transmission_rate, recovery_rate, mortality_rate, healthcare_capacity,
        lockdown_threshold, lockdown_strength, vaccination_coverage, vaccination_strategy, with_policies=False
    )
    msg = dbc.Alert("Policy analysis completed!", color="success")
    return {
        "history_with_policies": pickle.dumps(history_with_policies),
        "policy_activations": policy_activations,
        "history_without_policies": pickle.dumps(history_without_policies)
    }, msg

@app.callback(
    Output("policy-results", "children"),
    Input("policy-session", "data")
)
def display_policy_results(policy_data):
    if not policy_data or "history_with_policies" not in policy_data:
        return None
    history_with_policies = pickle.loads(policy_data["history_with_policies"])
    history_without_policies = pickle.loads(policy_data["history_without_policies"])
    policy_activations = policy_data["policy_activations"]
    # Metrics
    with_policies_df = pd.DataFrame(history_with_policies)
    without_policies_df = pd.DataFrame(history_without_policies)
    effectiveness = {
        'Peak Reduction': (without_policies_df['I'].max() - with_policies_df['I'].max()) / without_policies_df['I'].max() * 100 if without_policies_df['I'].max() > 0 else 0,
        'Death Reduction': (without_policies_df['D'].max() - with_policies_df['D'].max()) / without_policies_df['D'].max() * 100 if without_policies_df['D'].max() > 0 else 0,
        'Duration Reduction': (len(without_policies_df) - len(with_policies_df)) / len(without_policies_df) * 100 if len(without_policies_df) > 0 else 0
    }
    # Policy activation timeline
    activation_fig = None
    if policy_activations:
        activation_fig = go.Figure()
        for policy in set([p for act in policy_activations for p in act['policies']]):
            steps = [act['step'] for act in policy_activations if policy in act['policies']]
            activation_fig.add_trace(go.Scatter(x=steps, y=[1]*len(steps), mode='markers', name=policy, marker=dict(size=10)))
        activation_fig.update_layout(title="Policy Activation Timeline", xaxis_title="Time Step", yaxis_title="Policy", yaxis=dict(showticklabels=False), showlegend=True, template="plotly_white")
    # Disease spread comparison
    comp_fig = go.Figure()
    for state in ['S', 'I', 'R', 'D']:
        comp_fig.add_trace(go.Scatter(x=list(range(len(history_with_policies))), y=[h[state] for h in history_with_policies], name=f"{state} (With Policies)", line=dict(dash="solid")))
    for state in ['S', 'I', 'R', 'D']:
        comp_fig.add_trace(go.Scatter(x=list(range(len(history_without_policies))), y=[h[state] for h in history_without_policies], name=f"{state} (Without Policies)", line=dict(dash="dash")))
    comp_fig.update_layout(title="Policy Intervention Comparison", xaxis_title="Time Step", yaxis_title="Population Count", hovermode="x unified", template="plotly_white")
    # Effectiveness metrics
    metrics = [
        dbc.Col(dbc.Card([dbc.CardHeader("Peak Reduction"), dbc.CardBody(f"{effectiveness['Peak Reduction']:.1f}%")])),
        dbc.Col(dbc.Card([dbc.CardHeader("Death Reduction"), dbc.CardBody(f"{effectiveness['Death Reduction']:.1f}%")])),
        dbc.Col(dbc.Card([dbc.CardHeader("Duration Reduction"), dbc.CardBody(f"{effectiveness['Duration Reduction']:.1f}%")]))
    ]
    return html.Div([
        html.H4("Policy Analysis Results"),
        html.Hr(),
        html.H5("With Policies"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Peak Infected"), dbc.CardBody(str(with_policies_df['I'].max()))])),
            dbc.Col(dbc.Card([dbc.CardHeader("Total Deaths"), dbc.CardBody(str(with_policies_df['D'].max()))])),
            dbc.Col(dbc.Card([dbc.CardHeader("Final Recovered"), dbc.CardBody(str(with_policies_df['R'].iloc[-1]))])),
            dbc.Col(dbc.Card([dbc.CardHeader("Epidemic Duration"), dbc.CardBody(str(len(with_policies_df)))])),
        ]),
        html.H5("Without Policies", className="mt-4"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Peak Infected"), dbc.CardBody(str(without_policies_df['I'].max()))])),
            dbc.Col(dbc.Card([dbc.CardHeader("Total Deaths"), dbc.CardBody(str(without_policies_df['D'].max()))])),
            dbc.Col(dbc.Card([dbc.CardHeader("Final Recovered"), dbc.CardBody(str(without_policies_df['R'].iloc[-1]))])),
            dbc.Col(dbc.Card([dbc.CardHeader("Epidemic Duration"), dbc.CardBody(str(len(without_policies_df)))])),
        ]),
        html.Hr(),
        html.H4("Policy Effectiveness"),
        dbc.Row(metrics),
        html.Hr(),
        html.H4("Policy Activation Timeline"),
        dcc.Graph(figure=activation_fig, config={"displayModeBar": True, "responsive": True}) if activation_fig else html.Div(),
        html.Hr(),
        html.H4("Disease Spread Comparison"),
        dcc.Graph(figure=comp_fig, config={"displayModeBar": True, "responsive": True})
    ])

# --- Model Comparison Callbacks ---
def run_modelcomp_simulation(num_nodes, network_model, model_type, params, num_steps=100):
    G = generate_social_network(num_nodes, network_model)
    model = DiseaseModel(G, model_type, params)
    num_initial_infections = int(num_nodes * 0.05)
    nodes_to_infect = np.random.choice(list(G.nodes()), size=num_initial_infections, replace=False)
    for node in nodes_to_infect:
        model.states[node] = DiseaseState.INFECTED
        model.time_in_state[node] = 0
    history = []
    for _ in range(num_steps):
        state_counts = model.step()
        history.append(state_counts)
    return history

@app.callback(
    Output("modelcomp-session", "data"),
    Output("modelcomp-message", "children"),
    Input("run-modelcomp", "n_clicks"),
    State("mc-num-nodes", "value"),
    State("mc-network-model", "value"),
    State("mc-selected-models", "value"),
    State("mc-transmission-rate", "value"),
    State("mc-recovery-rate", "value"),
    State("mc-mortality-rate", "value"),
    State("mc-latency-period", "value"),
    State("mc-immunity-duration", "value"),
    State("mc-healthcare-capacity", "value"),
    prevent_initial_call=True
)
def run_modelcomp(n_clicks, num_nodes, network_model, selected_models, transmission_rate, recovery_rate, mortality_rate, latency_period, immunity_duration, healthcare_capacity):
    if n_clicks == 0 or not selected_models:
        return dash.no_update, dbc.Alert("Please select at least one model to compare.", color="warning")
    results = {}
    for model_type in selected_models:
        histories = []
        for _ in range(5):  # Run 5 simulations per model
            params = DiseaseParameters(
                transmission_rate=transmission_rate,
                recovery_rate=recovery_rate,
                mortality_rate=mortality_rate,
                latency_period=latency_period,
                immunity_duration=immunity_duration,
                healthcare_capacity=healthcare_capacity
            )
            history = run_modelcomp_simulation(num_nodes, network_model, model_type, params)
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
    msg = dbc.Alert("Model comparison completed!", color="success")
    return {"results": pickle.dumps(results)}, msg

@app.callback(
    Output("modelcomp-results", "children"),
    Input("modelcomp-session", "data")
)
def display_modelcomp_results(modelcomp_data):
    if not modelcomp_data or "results" not in modelcomp_data:
        return None
    results = pickle.loads(modelcomp_data["results"])
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
    # Create comparison plot
    fig = go.Figure()
    for model, result in results.items():
        # Calculate average history
        avg_history = []
        for step in range(100):
            step_counts = {}
            for state in ['S', 'E', 'I', 'R', 'D']:
                counts = [history[step].get(state, 0) for history in result['histories'] if step < len(history)]
                step_counts[state] = np.mean(counts) if counts else 0
            avg_history.append(step_counts)
        # Add traces for each state
        for state in ['S', 'E', 'I', 'R', 'D']:
            if state in avg_history[0]:
                fig.add_trace(go.Scatter(
                    x=list(range(len(avg_history))),
                    y=[h[state] for h in avg_history],
                    name=f"{state} ({model})",
                    line=dict(dash="solid" if model == "SIR" else "dot" if model == "SEIR" else "dash")
                ))
    fig.update_layout(title="Model Comparison", xaxis_title="Time Step", yaxis_title="Population Count", hovermode="x unified")
    return html.Div([
        html.H4("Model Comparison Results"),
        dbc.Table.from_dataframe(comparison_df.round(3), striped=True, bordered=True, hover=True),
        html.Hr(),
        dcc.Graph(figure=fig, config={"displayModeBar": True, "responsive": True})
    ])

if __name__ == "__main__":
    app.run_server(debug=True) 