import matplotlib.pyplot as plt
import networkx as nt
from Graph_1.creation_graph import Graph

gr = Graph()
gr.edge_to_add('A', 'B')
gr.edge_to_add('A', 'C')
gr.edge_to_add('B', 'D')
gr.edge_to_add('C', 'D')
gr.edge_to_add('C', 'F')
gr.edge_to_add('C', 'E')
gr.edge_to_add('D', 'E')
gr.edge_to_add('D', 'G')
gr.edge_to_add('E', 'H')
gr.edge_to_add('E', 'I')
gr.edge_to_add('E', 'F')
gr.edge_to_add('E', 'G')
gr.edge_to_add('G', 'J')
gr.edge_to_add('G', 'I')
gr.edge_to_add('H', 'I')
gr.edge_to_add('H', 'J')

gr.print_matrix_adjacency(gr.create_matrix_adjacency(),sorted(gr.total_graph.keys()))
print(gr.create_matrix_incidence())
gr.list_of_adjacency()
gr.visual()