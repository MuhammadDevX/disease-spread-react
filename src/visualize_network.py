import networkx as nx
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any, Optional, List
from src.disease_model import DiseaseState

def create_network_figure(
    G: nx.Graph,
    states: Dict[int, DiseaseState],
    infection_sources: Optional[Dict[int, int]] = None,
    layout: Optional[str] = "spring",
    **kwargs
) -> go.Figure:
    """
    Create an interactive network visualization using Plotly.
    
    Args:
        G: NetworkX graph
        states: Dictionary mapping node IDs to their disease states
        infection_sources: Dictionary mapping infected nodes to their source nodes
        layout: Layout algorithm to use ('spring', 'kamada_kawai', 'circular')
        **kwargs: Additional layout parameters
        
    Returns:
        plotly.graph_objects.Figure: Interactive network visualization
    """
    # Get node positions based on layout
    if layout == "spring":
        pos = nx.spring_layout(G, **kwargs)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G, **kwargs)
    elif layout == "circular":
        pos = nx.circular_layout(G, **kwargs)
    else:
        raise ValueError(f"Unknown layout algorithm: {layout}")
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Check if this edge represents an infection path
        is_infection_path = (
            infection_sources is not None and
            edge[1] in infection_sources and
            infection_sources[edge[1]] == edge[0]
        )
        
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(
                color='rgba(255, 0, 0, 0.8)' if is_infection_path else 'rgba(128, 128, 128, 0.3)',
                width=2 if is_infection_path else 0.5
            ),
            hoverinfo='text',
            text=f"Infection path: {edge[0]} → {edge[1]}" if is_infection_path else None
        ))
    
    # Define node colors and sizes based on state
    node_colors = {
        DiseaseState.SUSCEPTIBLE: 'rgba(0, 0, 255, 0.8)',    # Blue
        DiseaseState.EXPOSED: 'rgba(255, 255, 0, 0.8)',      # Yellow
        DiseaseState.INFECTED: 'rgba(255, 0, 0, 0.8)',       # Red
        DiseaseState.RECOVERED: 'rgba(0, 255, 0, 0.8)',      # Green
        DiseaseState.DEAD: 'rgba(0, 0, 0, 0.8)'             # Black
    }
    
    node_sizes = {
        DiseaseState.SUSCEPTIBLE: 10,
        DiseaseState.EXPOSED: 12,
        DiseaseState.INFECTED: 15,
        DiseaseState.RECOVERED: 10,
        DiseaseState.DEAD: 8
    }
    
    # Add nodes for each state
    for state in DiseaseState:
        nodes = [node for node, s in states.items() if s == state]
        if nodes:
            x = [pos[node][0] for node in nodes]
            y = [pos[node][1] for node in nodes]
            
            # Create hover text with infection source information
            hover_texts = []
            for node in nodes:
                text = f"Node {node}<br>State: {state.value}"
                if infection_sources and node in infection_sources:
                    text += f"<br>Infected by: {infection_sources[node]}"
                hover_texts.append(text)
            
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(
                    size=node_sizes[state],
                    color=node_colors[state],
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                name=state.value,
                hoverinfo='text',
                text=hover_texts
            ))
    
    # Update layout
    fig.update_layout(
        showlegend=True,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(
            title="Disease States",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_time_series_plot(
    history: List[Dict[str, int]],
    title: str = "Population State Over Time"
) -> go.Figure:
    """
    Create a time series plot of population states.
    
    Args:
        history: List of dictionaries containing state counts over time
        title: Plot title
        
    Returns:
        plotly.graph_objects.Figure: Time series visualization
    """
    fig = go.Figure()
    
    # Convert history to DataFrame for easier plotting
    import pandas as pd
    df = pd.DataFrame(history)
    
    # Add traces for each state
    for column in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[column],
            name=column,
            mode='lines+markers'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time Step",
        yaxis_title="Population Count",
        hovermode="x unified"
    )
    
    return fig

def create_heatmap(
    G: nx.Graph,
    states: Dict[int, DiseaseState],
    grid_size: int = 50
) -> go.Figure:
    """
    Create a heatmap visualization of infection density.
    
    Args:
        G: NetworkX graph
        states: Dictionary mapping node IDs to their disease states
        grid_size: Size of the grid for the heatmap
        
    Returns:
        plotly.graph_objects.Figure: Heatmap visualization
    """
    # Get node positions
    pos = nx.spring_layout(G)
    
    # Create grid
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Calculate infection density
    Z = np.zeros((grid_size, grid_size))
    for node, state in states.items():
        if state == DiseaseState.INFECTED:
            x_pos, y_pos = pos[node]
            i = int(x_pos * (grid_size - 1))
            j = int(y_pos * (grid_size - 1))
            Z[j, i] += 1
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=Z,
        x=x,
        y=y,
        colorscale='Reds',
        showscale=True
    ))
    
    # Update layout
    fig.update_layout(
        title="Infection Density Heatmap",
        xaxis_title="X Position",
        yaxis_title="Y Position"
    )
    
    return fig

def create_animated_infection_visualization(
    G: nx.Graph,
    states: Dict[int, DiseaseState],
    infection_sources: Dict[int, int],
    layout: Optional[str] = "spring",
    **kwargs
) -> go.Figure:
    """
    Create an animated visualization showing how infections spread through the network.
    
    Args:
        G: NetworkX graph
        states: Dictionary mapping node IDs to their disease states
        infection_sources: Dictionary mapping infected nodes to their source nodes
        layout: Layout algorithm to use
        **kwargs: Additional layout parameters
        
    Returns:
        plotly.graph_objects.Figure: Animated network visualization
    """
    # Get node positions
    if layout == "spring":
        pos = nx.spring_layout(G, **kwargs)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G, **kwargs)
    elif layout == "circular":
        pos = nx.circular_layout(G, **kwargs)
    else:
        raise ValueError(f"Unknown layout algorithm: {layout}")
    
    # Create figure
    fig = go.Figure()
    
    # Define colors and sizes
    node_colors = {
        DiseaseState.SUSCEPTIBLE: 'rgba(0, 0, 255, 0.8)',    # Blue
        DiseaseState.EXPOSED: 'rgba(255, 255, 0, 0.8)',      # Yellow
        DiseaseState.INFECTED: 'rgba(255, 0, 0, 0.8)',       # Red
        DiseaseState.RECOVERED: 'rgba(0, 255, 0, 0.8)',      # Green
        DiseaseState.DEAD: 'rgba(0, 0, 0, 0.8)'             # Black
    }
    
    node_sizes = {
        DiseaseState.SUSCEPTIBLE: 10,
        DiseaseState.EXPOSED: 12,
        DiseaseState.INFECTED: 15,
        DiseaseState.RECOVERED: 10,
        DiseaseState.DEAD: 8
    }
    
    # Create frames for animation
    frames = []
    
    # Track infected nodes across frames
    infected_nodes = set()
    
    # First frame: initial state
    initial_data = []
    
    # Add edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    initial_data.append(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(color='rgba(128, 128, 128, 0.3)', width=0.5),
            hoverinfo='none'
        )
    )
    
    # Add nodes for each state
    for state in DiseaseState:
        nodes = [node for node, s in states.items() if s == state]
        if nodes:
            if state == DiseaseState.INFECTED:
                infected_nodes.update(nodes)
            x = [pos[node][0] for node in nodes]
            y = [pos[node][1] for node in nodes]
            initial_data.append(
                go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=dict(
                        size=node_sizes[state],
                        color=node_colors[state],
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    name=state.value,
                    hoverinfo='text',
                    text=[f"Node {node}<br>State: {state.value}" for node in nodes]
                )
            )
    
    frames.append(go.Frame(data=initial_data, name="initial"))
    
    # Create frames for each infection
    for infected, source in infection_sources.items():
        # First frame: show infection path
        path_frame = []
        
        # Add background edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        path_frame.append(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(color='rgba(128, 128, 128, 0.3)', width=0.5),
                hoverinfo='none',
                name="Network"
            )
        )
        
        # Add all previously infected nodes
        prev_infected = [node for node in infected_nodes]
        if prev_infected:
            x = [pos[node][0] for node in prev_infected]
            y = [pos[node][1] for node in prev_infected]
            path_frame.append(
                go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=dict(
                        size=node_sizes[DiseaseState.INFECTED],
                        color=node_colors[DiseaseState.INFECTED],
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    name="Infected",
                    hoverinfo='text',
                    text=[f"Node {node}<br>State: {DiseaseState.INFECTED.value}" for node in prev_infected]
                )
            )
        
        # Add infection path
        x_path = [pos[source][0], pos[infected][0]]
        y_path = [pos[source][1], pos[infected][1]]
        
        path_frame.append(
            go.Scatter(
                x=x_path,
                y=y_path,
                mode='lines',
                line=dict(color='rgba(255, 0, 0, 0.8)', width=3),
                hoverinfo='text',
                text=f"Infection path: {source} → {infected}",
                name="Infection Path"
            )
        )
        
        # Add the node about to be infected in blue
        path_frame.append(
            go.Scatter(
                x=[pos[infected][0]],
                y=[pos[infected][1]],
                mode='markers',
                marker=dict(
                    size=node_sizes[DiseaseState.INFECTED],
                    color=node_colors[DiseaseState.SUSCEPTIBLE],
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                name="New Infection",
                hoverinfo='text',
                text=[f"Node {infected}<br>State: About to be infected"]
            )
        )
        
        frames.append(go.Frame(data=path_frame, name=f"infection_{infected}_path"))
        
        # Create transition frames for the newly infected node
        for i in range(5):  # Create 5 transition frames
            transition_frame = path_frame.copy()
            
            # Update the transitioning node's color
            t = i / 4  # t goes from 0 to 1
            r = int(255 * t)  # Red component increases
            b = int(255 * (1 - t))  # Blue component decreases
            
            # Find and update the transitioning node
            for trace in transition_frame:
                if trace.name == "New Infection":
                    trace.marker.color = f'rgba({r}, 0, {b}, 0.8)'
                    trace.text = [f"Node {infected}<br>State: Transitioning"]
            
            frames.append(go.Frame(data=transition_frame, name=f"infection_{infected}_transition_{i}"))
        
        # Add final frame with the node fully red
        final_frame = path_frame.copy()
        
        # Update the newly infected node to red
        for trace in final_frame:
            if trace.name == "New Infection":
                trace.marker.color = node_colors[DiseaseState.INFECTED]
                trace.name = "Infected"
                trace.text = [f"Node {infected}<br>State: {DiseaseState.INFECTED.value}"]
        
        frames.append(go.Frame(data=final_frame, name=f"infection_{infected}_final"))
        
        # Add the newly infected node to our tracking set
        infected_nodes.add(infected)
    
    # Add initial data to figure
    fig.add_traces(initial_data)
    
    # Add frames to figure
    fig.frames = frames
    
    # Update layout
    fig.update_layout(
        showlegend=True,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(
            title="Disease States",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 400, "redraw": True}}]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": True}}]
                    )
                ],
                direction="left",
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top"
            )
        ]
    )
    
    return fig
