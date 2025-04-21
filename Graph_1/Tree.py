import networkx as nx
import matplotlib.pyplot as plt
class Tree:
    def __init__(self):
        self.total_tree = {}

    def vertex_to_add(self, vertex: str) -> None:
        if not isinstance(vertex, str):
            raise TypeError('Type of vertex should be string')
        if vertex not in self.total_tree:
            self.total_tree[vertex] = []
    def add_edge(self,vertex_beg:str,vertex_end:str)->None:
        if not isinstance(vertex_beg,str):
            raise TypeError('Type of vertex should be string')
        if not isinstance(vertex_end,str):
            raise TypeError('Type of vertex should be string')
        self.vertex_to_add(vertex_beg)
        self.vertex_to_add(vertex_end)
        if vertex_end not in self.total_tree[vertex_beg]:
            self.total_tree[vertex_beg].append(vertex_end)
        if vertex_beg not in self.total_tree[vertex_end]:
            self.total_tree[vertex_end].append(vertex_beg)
    def prufer_encode(self):


        if len(self.total_tree) < 2:
            return None

        degree = {vertex: 0 for vertex in self.total_tree}
        for vertex, neighbors in self.total_tree.items():
            degree[vertex] = len(neighbors)

        prufer_code = []


        for _ in range(len(self.total_tree) - 2):
            leaf = min((v for v, d in degree.items() if d == 1), key=lambda x: int(x), default=None)
            if leaf is None:
                break


            neighbor = min(self.total_tree[leaf], key=lambda x: int(x))

            prufer_code.append(neighbor)


            degree[neighbor] -= 1
            degree[leaf] = 0
            self.total_tree[leaf].remove(neighbor)
            self.total_tree[neighbor].remove(leaf)

        return prufer_code

    def prufer_decode(self, prufer_code):

        n = len(prufer_code) + 2
        vertices = [str(i) for i in range(n)]
        degree = {vertex: 1 for vertex in vertices}
        for vertex in prufer_code:
            degree[vertex] += 1

        self.total_tree = {vertex: [] for vertex in vertices}

        for i in range(n - 2):
            leaf = min((v for v, d in degree.items() if d == 1), key=lambda x: int(x))
            neighbor = prufer_code[i]
            self.total_tree[leaf].append(neighbor)
            self.total_tree[neighbor].append(leaf)
            degree[leaf] -= 1
            degree[neighbor] -= 1

        last_nodes = [v for v, d in degree.items() if d == 1]
        self.total_tree[last_nodes[0]].append(last_nodes[1])
        self.total_tree[last_nodes[1]].append(last_nodes[0])
        return self.total_tree
    def visial_1(self):
        G = nx.Graph()


        for vertex, edges in self.total_tree.items():
            G.add_node(vertex)
            for adjacent, weight in edges.items():
                G.add_edge(vertex, adjacent, weight=weight)


        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'weight')

        plt.figure(figsize=(10, 8))


        nx.draw(G, pos, with_labels=True, node_color='red', node_size=2000, font_size=16)


        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.title("Граф с весами", fontsize=20)
        plt.show()



tree = Tree()
tree.add_edge('0', '1')
tree.add_edge('0', '2')
tree.add_edge('1', '3')
tree.add_edge('1', '4')
tree.add_edge('2', '5')
tree.add_edge('2', '6')
tree.add_edge('3', '7')
tree.add_edge('4', '8')
tree.add_edge('5', '9')
a = tree.prufer_encode()
print("Кодированное дерево",a)
decoded_tree = tree.prufer_decode(a)
print("Декодированное дерево:", decoded_tree)




