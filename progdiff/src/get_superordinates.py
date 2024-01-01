from nltk.corpus import wordnet as wn
from ordered_set import OrderedSet
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np


def load_childes_vocab_list():
    vocab_set = OrderedSet(['footwear', 'shoe'])
    # vocab_set = {'dog', 'canine', 'wolf', 'shoe', 'hose', 'footwear'}
    vocab_index_dict = {}
    for index, word in enumerate(vocab_set):
        vocab_index_dict[word] = index
    return vocab_index_dict


def load_category_file(file_path):
    vocab_index_dict = {}
    i = 0
    with open(file_path) as f:
        for line in f:
            data = (line.strip().strip('\n').strip()).split(',')
            vocab_index_dict[data[0]] = i
            i += 1
    return vocab_index_dict


def filter_grammatical_category(vocab_index_dict, category):
    filtered_vocab_index_dict = {}
    i = 0
    for word in vocab_index_dict:
        for synset in wn.synsets(word):
            if synset.pos() == category:  # Check if the synset represents a noun
                if word in [lemma.name() for lemma in synset.lemmas()]:
                    filtered_vocab_index_dict[word] = i
                    i += 1
                    break  # Break if you only need to add the word once
    return filtered_vocab_index_dict


def get_wordnet_data(vocab_index_dict, grammatical_category):
    """
    Processes a dictionary of words and their frequencies to extract WordNet data.

    Args:
        vocab_index_dict (dict): A dictionary where keys are words and values are frequencies.
        grammatical_category (string): a string indicating the grammatical category to include, None=all

    Returns:
        set: A set of synset objects containing WordNet data for the given words.
    """
    synset_network = nx.DiGraph()

    for word in vocab_index_dict:
        # Get all synsets for the word
        for synset in wn.synsets(word):
            # Check grammatical category if specified
            if grammatical_category is None or synset.pos() == grammatical_category:
                if len(synset.hypernyms()) > 0:
                    synset_network.add_node(synset.name(), synset=synset)

    return synset_network


def add_hypernyms(node, network):
    synset = network.nodes[node]['synset']
    for hypernym in synset.hypernyms():
        hypernym_name = hypernym.name()
        if hypernym_name not in network:
            network.add_node(hypernym_name, synset=hypernym)
            add_hypernyms(hypernym_name, network)  # Recursive call
        network.add_edge(hypernym_name, node)


def add_parents(network):
    """
    Recursively adds hypernyms of each synset in the network.

    Args:
        network (DiGraph): A directed graph of synset objects.

    Returns:
        DiGraph: The updated network with hypernyms added.
    """
    for node in list(network.nodes):
        add_hypernyms(node, network)

    return network


def get_all_children(graph, node, children=None):
    """
    Recursively get all descendants of a node in a directed graph.

    Args:
        graph (nx.DiGraph): The directed graph.
        node: The node whose descendants you want to find.
        children (set): A set to accumulate descendants.

    Returns:
        set: A set of all descendants of the node.
    """
    if children is None:
        children = set()

    if node not in graph:
        return children

    for child in graph.successors(node):
        if child not in children:
            children.add(child)
            get_all_children(graph, child, children)

    return children


def plot_network(network):
    pos = graphviz_layout(network, prog='dot')
    # Draw the graph
    nx.draw(network, pos, with_labels=True, node_color='lightblue', arrows=True)
    plt.show()


def print_category_sizes(network):
    node_children_counts = {}

    # Calculate the number of children for each node
    for node in network.nodes:
        synset = network.nodes[node]['synset']
        children = get_all_children(network, node)
        node_children_counts[synset.name()] = len(children)

    # Sort the nodes by the number of children in descending order
    sorted_nodes = sorted(node_children_counts.items(), key=lambda x: x[1], reverse=True)

    # Write the sorted data to the file
    with open("../../results/category_sizes.csv", "w") as f:
        for synset_name, count in sorted_nodes:
            if count > 0:
                f.write(f"{synset_name},{count}\n")


def get_lemma_list(node, network):
    lemma_list = []
    synset = network.nodes[node]['synset']
    for lemma in synset.lemma_names():
        lemma_list.append(lemma)
    return lemma_list


def test_categories(network, vocab_index_dict):

    synset_category_list = []
    vocab_list = list(vocab_index_dict.keys())

    for node in network.nodes:
        category_array = np.zeros([len(vocab_index_dict)])
        lemma_list = get_lemma_list(node, network)

        children = get_all_children(network, node)
        for child in children:
            lemma_list += get_lemma_list(child, network)

        for lemma in lemma_list:
            if lemma.lower() in vocab_index_dict:
                category_array[vocab_index_dict[lemma.lower()]] = 1

        synset = network.nodes[node]['synset']
        synset_category_list.append([category_array.sum(), synset.name(), category_array])

    sorted_list = sorted(synset_category_list, key=lambda x: x[0], reverse=True)
    for i in range(len(sorted_list)):
        if sorted_list[i][0] > 10:
            print(sorted_list[i][1])
            for j in range(len(vocab_index_dict)):
                if sorted_list[i][2][j] == 1:
                    print(f"    {vocab_list[j]}")




def main():
    """
    Main function to demonstrate the functionality of the get_wordnet_data function.
    """
    grammatical_category = 'n'
    vocab_file_path = "../../input_data/categories.csv"

    # get the vocab dict and filter by grammatical category
    vocab_index_dict = load_category_file(vocab_file_path)
    vocab_index_dict = filter_grammatical_category(vocab_index_dict, grammatical_category)

    # Get synset data for the words in the vocab dict, and all their parents
    synset_network = get_wordnet_data(vocab_index_dict, grammatical_category)
    synset_network = add_parents(synset_network)

    test_categories(synset_network, vocab_index_dict)
    # category_matrix = create_category_matrix(vocab_set, synset_network)

    # Print the lemma frequencies and superordinate lists for each WordSense object
    # for synset in synset_network:
    #     print(synset.lemma_names(), synset.hypernyms())
    #
    # for i, word in enumerate(vocab_set):
    #     print(word, category_matrix[i, :])


if __name__ == '__main__':
    main()
