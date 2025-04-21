import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from Struct.Min_heap import Min_Heap
import sys
import pandas as pd
from tabulate import tabulate


class Graph_with_weight(object):
    def __init__(self):
        self.total_graph = {}

    def vertex_to_add(self, vertex: str) -> None:
        if not isinstance(vertex, str):
            raise TypeError('Type of vertex should be string')
        if vertex not in self.total_graph:
            self.total_graph[vertex] = {}

    def edge_to_add(self, vertex_beg: str, vertex_end: str, weight: int) -> None:
        if not isinstance(vertex_beg, str):
            raise TypeError('Type of vertex should be string')
        if not isinstance(vertex_end, str):
            raise TypeError('Type of vertex should be string')
        if not isinstance(weight, int):
            raise TypeError('Type of vertex should be int')
        self.vertex_to_add(vertex_beg)
        self.vertex_to_add(vertex_end)
        self.total_graph[vertex_beg][vertex_end] = weight
        self.total_graph[vertex_end][vertex_beg] = weight

    def get_ver_from_index(self, i: int) -> list:
        if not isinstance(i, int):
            raise TypeError('Type of vertex should be int')
        return list(self.total_graph.keys())[i]

    def get_near_vertexs(self, vertex: str) -> list:
        if not isinstance(vertex, str):
            raise TypeError('Type of vertex should be string')
        if vertex in self.total_graph:
            return list(self.total_graph[vertex].keys())

    def vertex_to_remove(self, vertex: str) -> None:
        if not isinstance(vertex, str):
            raise TypeError('Type of vertex should be string')
        if vertex in self.total_graph:
            for v in list(self.total_graph[vertex].keys()):
                del self.total_graph[v][vertex]  # удалеям вес
            # print(self.total_graph)
            del self.total_graph[vertex]

    def remove_edge_for_graph(self, vertex_beg: str, vertex_end: str) -> None:
        if not isinstance(vertex_beg, str):
            raise TypeError('Type of vertex should be string')
        if not isinstance(vertex_end, str):
            raise TypeError('Type of vertex should be string')
        if vertex_beg in self.total_graph and vertex_end in self.total_graph[vertex_beg]:
            del self.total_graph[vertex_beg][vertex_end]
            del self.total_graph[vertex_end][vertex_beg]
            print(self.total_graph)

    def visial_1(self):
        G = nx.Graph()

        # Добавляем вершины и ребра в граф NetworkX
        for vertex, edges in self.total_graph.items():
            G.add_node(vertex)
            for adjacent, weight in edges.items():
                G.add_edge(vertex, adjacent, weight=weight)

        # Рисуем граф
        pos = nx.spring_layout(G)  # Определяем позиции узлов
        edge_labels = nx.get_edge_attributes(G, 'weight')  # Получаем веса ребер

        plt.figure(figsize=(10, 8))

        # Рисуем узлы и ребра
        nx.draw(G, pos, with_labels=True, node_color='red', node_size=2000, font_size=16)

        # Рисуем веса ребер
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.title("Граф с весами", fontsize=20)
        plt.show()

    def visial_2(self):
        G = nx.Graph()

        # Добавляем вершины и ребра в граф NetworkX
        for vertex, edges in self.total_graph.items():
            G.add_node(vertex)
            for adjacent, weight in edges.items():
                G.add_edge(vertex, adjacent, weight=weight)

        # Получаем позиции узлов для визуализации
        pos = nx.spring_layout(G)

        # Создаем фигуру Plotly
        edge_x = []
        edge_y = []
        edge_weights = []

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)  # Разрыв между ребрами
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)  # Разрыв между ребрами
            edge_weights.append(edge[2]['weight'])

        node_x = []
        node_y = []
        for node in pos:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        # Создаем график
        fig = go.Figure()

        # Добавляем ребра
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'))
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=list(G.nodes()),
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                color='blue',
                line_width=2)))

        fig.update_layout(
            title='Граф с весами',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False))

        fig.show()

    def dijkstra(self, start, end):
        distances = {vertex: float('infinity') for vertex in self.total_graph}  # Использовать self.total_graph
        distances[start] = 0
        previous_vertices = {vertex: None for vertex in self.total_graph}  # Использовать self.total_graph
        priority_queue = Min_Heap()
        priority_queue.push(start, 0)

        while not priority_queue.is_heap_emty():
            current_distance, current_vertex = priority_queue.pop()

            if current_distance > distances[current_vertex]:
                continue

            for neighbor, weight in self.total_graph.get(current_vertex, {}).items():
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_vertices[neighbor] = current_vertex
                    priority_queue.push(neighbor, distance)

        path = []
        current = end
        while current is not None:
            path.insert(0, current)
            current = previous_vertices[current]

        return distances.get(end), path
    def show(self,ver_beg:str):
        results = {}
        for end in self.total_graph:  # Для каждой конечной вершины
            distance, path = self.dijkstra(ver_beg, end)  # Используем существующий dijkstra
            results[end] = {'ver_beg':ver_beg,'weight': distance, 'path': path,'ver_end':path[-1]}
        return results

    def prim_mst(self):

        if not self.total_graph:
            return None

        num_vertices = len(self.total_graph)
        vertices = list(self.total_graph.keys())


        key = [sys.maxsize] * num_vertices
        parent = [None] * num_vertices
        key[0] = 0
        mstSet = [False] * num_vertices

        for _ in range(num_vertices):
            min_key = sys.maxsize
            u = -1

            for v in range(num_vertices):
                if key[v] < min_key and not mstSet[v]:
                    min_key = key[v]
                    u = v

            if u == -1:
                break

            mstSet[u] = True

            for v in range(num_vertices):
                neighbor_vertex = vertices[v]
                current_vertex = vertices[u]
                if neighbor_vertex in self.total_graph[current_vertex] and not mstSet[v]:
                    weight = self.total_graph[current_vertex][neighbor_vertex]
                    if weight < key[v]:
                        key[v] = weight
                        parent[v] = u


        mst = []
        for i in range(1, num_vertices):
            if parent[i] is not None:
                mst.append((vertices[parent[i]], vertices[i], self.total_graph[vertices[parent[i]]][vertices[i]]))

        return mst

    def generate_permutations(self, vertices, start=0):
        if start == len(vertices) - 1:
            yield vertices[:]
        for i in range(start, len(vertices)):
            vertices[start], vertices[i] = vertices[i], vertices[start]
            yield from self.generate_permutations(vertices, start + 1)
            vertices[start], vertices[i] = vertices[i], vertices[start]

    def greedy_algorithm(self) -> tuple:
        vertices = list(self.total_graph.keys())
        min_cycle = None
        min_weight = float('inf')
        for perm in self.generate_permutations(vertices):
            current_weight = 0
            valid_cycle = True

            for i in range(len(perm)):
                start_vertex = perm[i]
                end_vertex = perm[(i + 1) % len(perm)]
                if end_vertex in self.total_graph[start_vertex]:
                    current_weight += self.total_graph[start_vertex][end_vertex]
                else:
                    valid_cycle = False
                    break


            if valid_cycle and current_weight < min_weight:
                min_weight = current_weight
                min_cycle = perm

        return (min_cycle,min_weight)
    def visial_loop(self,arr:list):
        g = nx.Graph()
        for ver in range(len(arr)):
            g.add_edge(arr[ver],arr[(ver + 1) % len(arr)])
        pos = nx.spring_layout(g)
        nx.draw(g,pos,with_labels=True, node_color='red', node_size=1000, font_size=16)
        plt.title("Визуализация")
        plt.show()



graph = Graph_with_weight()
graph.edge_to_add('1', '2', 1)
graph.edge_to_add('1', '5', 4)
graph.edge_to_add('1', '3', 9)
graph.edge_to_add('2', '3', 4)
graph.edge_to_add('2', '4', 1)
graph.edge_to_add('3', '4', 2)
graph.edge_to_add('3', '6', 3)
graph.edge_to_add('3', '5', 2)
graph.edge_to_add('4', '5', 1)
graph.edge_to_add('4', '7', 5)
graph.edge_to_add('5', '6', 2)
graph.edge_to_add('5', '9', 3)
graph.edge_to_add('5', '8', 3)
graph.edge_to_add('5', '7', 7)
graph.edge_to_add('6', '9', 1)
graph.edge_to_add('7', '8', 1)
graph.edge_to_add('7', '10', 4)
graph.edge_to_add('8', '9', 1)
graph.edge_to_add('8', '10', 4)
graph.edge_to_add('9', '10', 2)
print(graph.dijkstra('1','5'))
'''
str1 = str(input())
if not isinstance(str1,str):
    raise TypeError('Type should be string')

res = graph.show(str1)
df = pd.DataFrame.from_dict(res, orient='index')
print(tabulate(df,headers="keys", tablefmt="grid", showindex=False))
graph.visial_1()
graph.visial_2()

a = graph.prim_mst()
beg = []
end = []
weight = []
if a:
    for i,j,k in a:
        beg.append(i)
        end.append(j)
        weight.append(k)
        print(f"{i} -- {j} : {k}")
def visual1_prima(start_nodes, end_nodes, weights):
    graph = nx.Graph()
    for i in range(len(start_nodes)):
        graph.add_edge(start_nodes[i], end_nodes[i], weight=weights[i])

    pos = nx.spring_layout(graph)  # Расположение вершин (можно изменить)
    nx.draw(graph, pos, with_labels=True, node_color='red', node_size=500, font_size=10)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Граф с весами")
    plt.show()

visual1_prima(beg,end,weight)'''







