import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components

def visualize_social_network(G, title="Social Network"):
    pos = nx.spring_layout(G, seed=42)
    status_colors = {'healthy': 'green', 'infected': 'red', 'recovered': 'blue', 'dead': 'gray'}
    colors = [status_colors[G.nodes[n]['status']] for n in G.nodes]

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(G, pos, node_color=colors, with_labels=False, node_size=30, ax=ax)
    ax.set_title(title)
    return fig

def display_interactive_network(G, height="600px"):
    net = Network(height=height, bgcolor="#222222", font_color="white")
    status_colors = {'healthy': 'green', 'infected': 'red', 'recovered': 'blue', 'dead': 'gray'}

    for node in G.nodes:
        net.add_node(node, color=status_colors[G.nodes[node]['status']])
    for edge in G.edges:
        net.add_edge(edge[0], edge[1])

    net.force_atlas_2based()
    net.show("network.html")

    HtmlFile = open("network.html", 'r', encoding='utf-8')
    components.html(HtmlFile.read(), height=650, scrolling=True)
