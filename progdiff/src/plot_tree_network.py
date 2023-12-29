import networkx as nx
import matplotlib.pyplot as plt


def create_network(node_list, directed=False):

    if directed:
        network = nx.DiGraph()
    else:
        network = nx.Graph()  # Create an undirected graph

    for node in node_list:
        network.add_node(node[0])

    for node in node_list:
        for parent in node[1]:
            network.add_edge(parent, node[0])


def plot_network(network, node_label_list=None):
    pos = nx.spring_layout(network)  # Define the layout
    nx.draw(network, pos, with_labels=False)  # Draw the graph without default labels

    if node_label_list is not None:
        nx.draw_networkx_labels(network, pos, labels=node_label_list)
    else:
        nx.draw(network, pos, with_labels=True)
    plt.show()
