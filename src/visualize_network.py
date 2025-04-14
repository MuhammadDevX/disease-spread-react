import networkx as nx
import matplotlib.pyplot as plt

def visualize_social_network(G, title="Social Network"):
    """
    Visualizes the social network with color-coded nodes.
    """
    pos = nx.spring_layout(G)
    statuses = nx.get_node_attributes(G, 'status')
    colors = ['red' if statuses[n] == 'infected' else 'green' for n in G.nodes]

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_color=colors, with_labels=False, node_size=30)
    plt.title(title)
    plt.show()
