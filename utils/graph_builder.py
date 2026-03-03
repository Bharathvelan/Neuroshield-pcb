# graph_builder.py
# Converts PCB design data into a graph structure for our AI

import networkx as nx
import numpy as np
import sys
sys.path.append('data')

def build_pcb_graph(pcb_row):
    """
    Takes one PCB design (one row of our data)
    and converts it into a graph.

    Nodes = parts of the PCB (trace, via, decap, ground)
    Edges = connections between them
    """

    G = nx.Graph()

    # --- Add Nodes (components of the PCB) ---

    # Node 1: The main signal trace
    G.add_node('trace',
               width=pcb_row['trace_width_mm'],
               length=pcb_row['trace_length_mm'],
               frequency=pcb_row['frequency_mhz'],
               node_type=0)

    # Node 2: The ground plane
    G.add_node('ground',
               distance=pcb_row['ground_distance_mm'],
               node_type=1)

    # Node 3: Stitching vias
    G.add_node('via',
               count=pcb_row['stitching_vias'],
               node_type=2)

    # Node 4: Decoupling capacitor
    G.add_node('decap',
               distance=pcb_row['decap_distance_mm'],
               node_type=3)

    # --- Add Edges (connections between components) ---

    G.add_edge('trace', 'ground',
               coupling=1.0 / pcb_row['ground_distance_mm'])

    G.add_edge('trace', 'via',
               coupling=float(pcb_row['stitching_vias']))

    G.add_edge('trace', 'decap',
               coupling=1.0 / pcb_row['decap_distance_mm'])

    G.add_edge('via', 'ground',
               coupling=1.0)

    return G


def graph_to_feature_vector(G):
    """
    Converts the graph into a simple list of numbers
    that our neural network can process.
    """
    features = []

    for node in G.nodes():
        node_data = G.nodes[node]
        for key, value in node_data.items():
            if isinstance(value, (int, float)):
                features.append(float(value))

    for edge in G.edges():
        edge_data = G.edges[edge]
        for key, value in edge_data.items():
            if isinstance(value, (int, float)):
                features.append(float(value))

    return np.array(features, dtype=np.float32)


# Test it immediately
if __name__ == "__main__":
    from sample_board import generate_pcb_samples

    df = generate_pcb_samples(10)

    # Build a graph for the first PCB design
    first_pcb = df.iloc[0]
    G = build_pcb_graph(first_pcb)

    print("PCB Graph Created!")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Nodes: {list(G.nodes())}")
    print(f"Edges: {list(G.edges())}")

    # Convert to feature vector
    features = graph_to_feature_vector(G)
    print(f"\nFeature vector length: {len(features)}")
    print(f"Feature vector: {features}")
