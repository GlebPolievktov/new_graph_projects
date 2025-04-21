import networkx as nx
import pandas as pd
from collections import deque
import numpy as np
from tabulate import tabulate
from Struct.Min_heap import Min_Heap
from Struct.Deque import Deque
import matplotlib.pyplot as plt

class Vertex(object):
    pass
class Orient_Graph_with_weight(object):
    def __init__(self):
        self.total_graph = {}
    def add_edge(self,ver1:str,ver2:str,weight)->None:
        if not isinstance(ver1,str):
            raise TypeError("Vertex should be string")
        if not isinstance(ver2,str):
            raise TypeError("Vertex should be string")
        if not isinstance(weight,int):
            raise TypeError("Weight should be int")
        if ver1 not in self.total_graph:
            self.total_graph[ver1] = {}
        self.total_graph[ver1][ver2] = weight
    def get_neighbors(self,ver:str)->dict:
        if not isinstance(ver,str):
            raise TypeError("Vertex should be string")
        return self.total_graph.get(ver,{})
    def get_weight(self,ver:str,end:str):
        if not isinstance(ver,str):
            raise TypeError("Vertex should be string")
        return self.total_graph.get(ver,{}).get(end,float('inf'))

    def remove_edge_for_graph(self, vertex_beg: str, vertex_end: str) -> None:
        if not isinstance(vertex_beg, str):
            raise TypeError('Type of vertex should be string')
        if not isinstance(vertex_end, str):
            raise TypeError('Type of vertex should be string')
        if vertex_beg in self.total_graph and vertex_end in self.total_graph[vertex_beg]:
            del self.total_graph[vertex_beg][vertex_end]
        else:
            raise ValueError("Error")
    def remove_ver(self,ver:str)->None:
        if not isinstance(ver, str):
            raise TypeError('Type of vertex should be string')
        if ver in self.total_graph:
            del self.total_graph[ver]
            for v in self.total_graph:
                if ver in self.total_graph[v]:
                    del self.total_graph[v][ver]
    def vis(self):
        G = nx.DiGraph()


        for vertex, edges in self.total_graph.items():
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

    def dijkstra(self, start, end):
        if not isinstance(start, str) or not isinstance(end, str):
            raise TypeError("Start and End vertices must be strings.")

        all_vertices = set(self.total_graph.keys())
        for neighbors in self.total_graph.values():
            all_vertices.update(neighbors.keys())

        if start not in all_vertices or end not in all_vertices:
            raise ValueError("Start or End vertex not in graph.")

        distances = {vertex: float('infinity') for vertex in all_vertices}
        distances[start] = 0
        previous_vertices = {vertex: None for vertex in all_vertices}
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

        path = deque()
        current = end
        while current is not None:
            path.appendleft(current)
            current = previous_vertices.get(current)

        return distances.get(end, float('inf')), list(path)



    def throughput_capacity(self, source: str, sink: str) -> int:


        if not isinstance(source, str) or not isinstance(sink, str):
            raise TypeError("Source and sink must be strings.")

        all_vertices = set(self.total_graph.keys())
        for neighbors in self.total_graph.values():
            all_vertices.update(neighbors.keys())

        if source not in all_vertices or sink not in all_vertices:
            raise ValueError("Source or sink vertex not in graph.")

        residual_graph = {vertex: {neighbor: weight for neighbor, weight in edges.items()} for vertex, edges in
                          self.total_graph.items()}
        max_flow = 0

        while True:
            path = self.find_path(residual_graph, source, sink)
            if path is None:
                break

            path_flow = float('inf')
            for i in range(len(path) - 1):
                path_flow = min(path_flow, residual_graph[path[i]].get(path[i + 1], 0))

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                residual_graph[u][v] -= path_flow
                residual_graph.setdefault(v, {})
                if u not in residual_graph[v]:
                    residual_graph[v][u] = 0
                residual_graph[v][u] += path_flow

            max_flow += path_flow

        return max_flow

    def find_path(self, graph, source, sink):

        queue = deque([(source, [source])])
        visited = set()

        while queue:
            (vertex, path) = queue.popleft()
            if vertex == sink:
                return path

            visited.add(vertex)
            for neighbor in graph.get(vertex, {}):
                if graph[vertex].get(neighbor, 0) > 0 and neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        return None

    def floyd_warshall(self):
        vertices = list(self.total_graph.keys())
        distance = {v: {u: float('inf') for u in vertices} for v in vertices}

        for v in vertices:
            distance[v][v] = 0
            for neighbor, weight in self.total_graph[v].items():
                distance[v][neighbor] = weight

        for k in vertices:
            for i in vertices:
                for j in vertices:
                    if distance[i][j] > distance[i][k] + distance[k][j]:
                        distance[i][j] = distance[i][k] + distance[k][j]
        return distance




gr = Orient_Graph_with_weight()
gr.add_edge('1','2',1)
gr.add_edge('2','3',4)
gr.add_edge('1','3',9)
gr.add_edge('4','2',1)
gr.add_edge('3','6',3)
gr.add_edge('4','3',2)
gr.add_edge('4','5',1)
gr.add_edge('3','5',2)
gr.add_edge('1','5',4)
gr.add_edge('5','6',2)
gr.add_edge('5','9',3)
gr.add_edge('6','9',1)
gr.add_edge('5','7',7)
gr.add_edge('7','4',5)
gr.add_edge('7','8',4)
gr.add_edge('8','9',1)
gr.add_edge('8','5',3)
gr.add_edge('8','10',4)
gr.add_edge('7','10',4)
gr.add_edge('9','10',2)

print(gr.dijkstra('1','6'))
print(gr.throughput_capacity('1','10'))
gr.vis()
s = gr.floyd_warshall()


for start in s:
    for end in s[start]:
        print(f"Кратчайшее расстояние от {start} до {end} равно {s[start][end]}")






