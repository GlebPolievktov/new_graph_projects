import networkx as nx
import pandas as pd
import numpy as np
from tabulate import tabulate
import plotly.graph_objects as go
from collections import deque
import pyqtgraph as pg
import networkx as nt
from Struct.Deque import Deque
from Struct.Stack import Stack
import matplotlib.pyplot as plt

class Vertex(object):
    pass

class Orient_Graph(object):
    def __init__(self):
        self.total_graph = {}

    def add_edge(self, ver1, ver2):
        '''ver1->ver2'''
        if not isinstance(ver1, str) or not isinstance(ver2, str):
            raise TypeError("Vertices must be strings")

        if ver1 not in self.total_graph:
            self.total_graph[ver1] = []
        self.total_graph[ver1].append(ver2)

    def get_near_vertexs(self, ver: str) -> list[str]:
        if not isinstance(ver, str):
            raise TypeError("Vertex should be string")
        if ver not in self.total_graph:
            return []
        else:
            return self.total_graph[ver]

    def remove_edge_for_graph(self, vertex_beg: str, vertex_end: str) -> None:
        if not isinstance(vertex_beg, str):
            raise TypeError('Type of vertex should be string')
        if not isinstance(vertex_end, str):
            raise TypeError('Type of vertex should be string')
        if vertex_beg in self.total_graph and vertex_end in self.total_graph[vertex_beg]:
            self.total_graph[vertex_beg].remove(vertex_end)
        else:
            raise ValueError("Edge should be in total_graph")

    def del_ver(self, ver: str)-> None:
         if not isinstance(ver, str):
            raise TypeError('Type of vertex should be string')
         if ver in self.total_graph:
             del self.total_graph[ver]
             for i in self.total_graph:
                 if ver in self.total_graph[i]:
                     self.total_graph[i].remove(ver)

    def count_ver(self):
        return list(self.total_graph.keys())

    def count_adg(self, ver: str):
        return list(self.total_graph[ver])

    def not_ver_adj(self, ver: str):
        if not isinstance(ver, str):
            raise TypeError("Vertex should be string")
        all_ver = set(self.total_graph.keys())
        all_adj_ver = set(self.get_near_vertexs(ver))
        tot = all_ver - all_adj_ver - {ver}
        tot = sorted(tot, reverse=False)
        return tot

    def list_of_adjacency(self):
        ln = len(self.total_graph) + 1
        num = [i for i in range(1, ln)]
        d1 = []
        for i in self.total_graph:
            d1.append(i)
        d2 = []
        for j in self.total_graph:
            d2.append(self.total_graph[j])
        data = {'Номер': num,
                'Вершина': d1,
                'Смежные вершины': d2}
        df = pd.DataFrame(data)
        print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))

    def search_in_width(self, start_vertex: str):
        if not isinstance(start_vertex, str):
            raise TypeError("Vertex must be a string")
        if start_vertex not in self.total_graph:
            raise ValueError("Start vertex not in graph")
        dq = Deque()
        visited = []
        dq.append(start_vertex)

        while not dq.is_empty() :
            vertex = dq.popleft()
            if vertex not in visited:
                visited.append(vertex)
                neighbors = self.get_near_vertexs(vertex)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        dq.append(neighbor)

        return visited
    def search_in_deep(self,ver_start:str):
        if not isinstance(ver_start,str):
            raise TypeError("Vertex should be string")
        if ver_start not in self.total_graph:
            raise ValueError("Start vertex not in graph")
        visited = []
        stack = Stack()
        stack.push(ver_start)

        while not stack.is_empty():
            vertex = stack.pop()
            if vertex not in visited:
                visited.append(vertex)
                neighbors = self.get_near_vertexs(vertex)
                for neighbor in reversed(neighbors):
                    if neighbor not in visited:
                        stack.push(neighbor)

        return visited
    def print_deep(self,ver):
        res = self.search_in_deep(ver)
        res = [res]
        data = {'Номер': 1,
                'Вершина': ver,
                'Поиск': res}
        df = pd.DataFrame(data)
        print("Поиск в длину")
        print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    def print_width(self,ver:str):
        res = self.search_in_width(ver)
        res = [res]
        data = {'Номер': 1,
                'Вершина': ver,
                'Поиск': res}
        df = pd.DataFrame(data)
        print("Поиск в ширину")
        print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    def find_all_ways_of_chain(self, node_start:str, node_end:str)->list:
        if not isinstance(node_start,str):
            raise TypeError('Type of vertex should be string')
        if not isinstance(node_end,str):
            raise TypeError('Type of vertex should be string')
        chain = []
        visit = set()
        t = [(node_start, [node_start])]
        while t:
            (vertex, p) = t.pop()
            if vertex not in visit:
                if vertex == node_end:
                    chain.append(p)
                else:
                    visit.add(vertex)
                    for i in self.total_graph.get(vertex, []):
                        if i not in p:
                            t.append((i, p + [i]))
                    try:
                        visit.remove(vertex)
                    except KeyError:
                        pass
        return chain
    def print_chain(self,mat:list)->list:
        num = [i for i in range(1,len(mat)+1)]
        node_1 = []
        node_2 = []
        for i in range(len(mat)):
            node_1.append(mat[0][0])
        for j in range(len(mat)):
            node_2.append(mat[0][-1])
        for i in range(len(mat)):
            del mat[i][0]
            del mat[i][-1]
        data = {'Номер': num,
                'Начало': node_1,
                'Цепь':mat,
                'Конец': node_2}
        df = pd.DataFrame(data)
        print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))

    def visial_1(self):
        g = nx.DiGraph()
        g.add_nodes_from(self.total_graph.keys())
        for v,near in self.total_graph.items():
            for n in near:
                g.add_edge(v,n)
        pos = nx.circular_layout(g)
        plt.figure(figsize=(10,8))
        nx.draw(g, pos, with_labels=True, node_size=1500, node_color="red", font_size=10, width=2, arrows=True)

        plt.title("Ориентированный граф")
        plt.show()




gr = Orient_Graph()
gr.add_edge('1','2')
gr.add_edge('2','3')
gr.add_edge('1','3')
gr.add_edge('4','2')
gr.add_edge('3','6')
gr.add_edge('4','3')
gr.add_edge('4','5')
gr.add_edge('3','5')
gr.add_edge('1','5')
gr.add_edge('5','6')
gr.add_edge('5','9')
gr.add_edge('6','9')
gr.add_edge('5','7')
gr.add_edge('7','4')
gr.add_edge('7','10')
gr.add_edge('7','8')
gr.add_edge('8','10')
gr.add_edge('8','9')
gr.add_edge('8','5')
gr.add_edge('9','10')
gr.add_edge('7','8')

gr.print_width('1')
gr.print_deep('1')
gr.print_chain(gr.find_all_ways_of_chain('1','10'))
gr.visial_1()
