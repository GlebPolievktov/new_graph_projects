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
print('Введите любые 2 цифры/числа ')
ver1 = str(input())
ver2 = str(input())
if not (type(ver1) and type(ver2)) == str:
    raise TypeError
gr.print_chain(gr.find_all_ways_of_chain(ver1,ver2))
gr.visual()