import pandas as pd
import numpy as np
from tabulate import tabulate
import networkx as nt
import plotly.graph_objects as go
import pyqtgraph as pg
from pyqtgraph import PlotWidget


import matplotlib.pyplot as plt

from Struct.Queue import Queue

class Vertex(object):
    pass
class Graph(object):
    def __init__(self):
        self.total_graph = {}
    def vertex_to_add(self,vertex:str)->None:
        if not isinstance(vertex,str):
            raise TypeError('Type of vertex should be string')
        if vertex not in self.total_graph:
            self.total_graph[vertex] = []
    def edge_to_add(self,vertex_beg:str,vertex_end:str)->None:
        if not isinstance(vertex_beg,str):
            raise TypeError('Type of vertex should be string')
        if not isinstance(vertex_end,str):
            raise TypeError('Type of vertex should be string')
        self.vertex_to_add(vertex_beg)
        self.vertex_to_add(vertex_end)
        if vertex_end not in self.total_graph[vertex_beg]:
            self.total_graph[vertex_beg].append(vertex_end)
        if vertex_beg not in self.total_graph[vertex_end]:
            self.total_graph[vertex_end].append(vertex_beg)
    def get_near_vertexs(self,vertex:str)->list[str]:
        if vertex in self.total_graph:
            return self.total_graph[vertex]
        else:
            return None # ну или пустой список без разницы -> []
    def getter(self,i:int)->str:
        if not isinstance(i,int):
            raise TypeError("Ratio should be integer")
        return (list(self.total_graph.keys())[i])
    def remove_edge_for_graph(self,vertex_beg:str,vertex_end:str)->None:
        if not isinstance(vertex_beg,str):
            raise TypeError('Type of vertex should be string')
        if not isinstance(vertex_end,str):
            raise TypeError('Type of vertex should be string')
        if vertex_beg in self.total_graph and vertex_end in self.total_graph:
            self.total_graph[vertex_beg] = [near for near in self.total_graph[vertex_beg] if near != vertex_end] #[B,C,D] становиться [C,D]
            self.total_graph[vertex_end] = [near for near in self.total_graph[vertex_end] if near != vertex_beg]
    def remove_vertex_for_graph(self,vertex)->None:
        if vertex in self.total_graph:
            del self.total_graph[vertex]
            for v in self.total_graph:
                self.total_graph[v] = [near for near in self.total_graph[v] if near != vertex]
    def not_adj_ver(self,ver:str):
        if ver not in self.total_graph:
            raise TypeError('Vertex should be ')
        all_ver = set(self.total_graph.keys())
        all_adj_ver = set(self.get_near_vertexs(ver))
        m = all_ver - all_adj_ver - {ver}
        m = sorted(m,reverse=True)
        return m
    def count(self):
        a = self.total_graph.keys()
        print(a)


    def visual(self)->None:
        g = nt.Graph()
        g.add_nodes_from(self.total_graph.keys())
        for ver,near in self.total_graph.items():
            for n in near:
                g.add_edge(ver, n)
        nt.draw(g, with_labels=True, node_size=1500, node_color="red", font_size=10, width=2)
        plt.show()
    def visual_new(self) -> None:
        # Создаем граф с помощью networkx
        g = nt.Graph()
        g.add_nodes_from(self.total_graph.keys())

        for ver, near in self.total_graph.items():
            for n in near:
                g.add_edge(ver, n)

        # Получаем позиции узлов графа
        pos = nt.spring_layout(g)

        # Извлекаем координаты узлов
        x_edges = []
        y_edges = []

        for edge in g.edges():
            x_edges.append(pos[edge[0]][0])
            x_edges.append(pos[edge[1]][0])
            x_edges.append(None)  # Для разрыва линии между парами вершин

            y_edges.append(pos[edge[0]][1])
            y_edges.append(pos[edge[1]][1])
            y_edges.append(None)  # Для разрыва линии между парами вершин

        # Создаем линии для рёбер графа
        edge_trace = go.Scatter(
            x=x_edges,
            y=y_edges,
            line=dict(width=0.5, color='black'),
            hoverinfo='none',
            mode='lines'
        )

        # Извлекаем координаты узлов
        x_nodes = []
        y_nodes = []

        for node in g.nodes():
            x_nodes.append(pos[node][0])
            y_nodes.append(pos[node][1])

        # Создаем точки для узлов графа
        node_trace = go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode='markers+text',
            text=list(g.nodes()),
            textposition="bottom center",
            marker=dict(
                showscale=False,
                color='red',
                size=10,
                line_width=2)
        )

        # Создаем фигуру и отображаем граф
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='<br>Граф<br>',
                            titlefont_size=16,
                            showlegend=True,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )

        fig.show()




    def visual3(self) -> None:

        g = nt.Graph()
        g.add_nodes_from(self.total_graph.keys())

        for ver, near in self.total_graph.items():
            for n in near:
                g.add_edge(ver, n)


        pos = nt.spring_layout(g)


        app = pg.mkQApp("Graph Visualization")
        plot_widget = PlotWidget()
        plot_widget.setWindowTitle("Graph Visualization")


        for edge in g.edges():
            x = [pos[edge[0]][0], pos[edge[1]][0]]
            y = [pos[edge[0]][1], pos[edge[1]][1]]
            plot_widget.plot(x, y, pen='k')  # 'k' для черного цвета


        for node in g.nodes():
            x = pos[node][0]
            y = pos[node][1]
            plot_widget.plot([x], [y], pen=None, symbol='o', symbolBrush='r', symbolSize=10)  # Красные узлы


            label = pg.TextItem(node, anchor=(0.5, 0.5))
            label.setPos(x, y)
            plot_widget.addItem(label)

        plot_widget.show()
        pg.exec()
    def create_matrix_adjacency(self)->list[str]:
        vr = sorted(self.total_graph.keys())
        ln = len(vr)
        matrix_adj = [[0] * ln for _ in range(ln)]
        dict_index = {ver:index for index,ver in enumerate(vr)}
        for i in self.total_graph:
            #print(i)
            for j in self.total_graph[i]:
                matrix_adj[dict_index[i]][dict_index[j]] = 1
        return matrix_adj
    def print_matrix_adjacency(self,matrix:list,vr:list)->None:
        print("Таблица смежности")
        for i in vr:
            print(f"   {i}",end="")
        print()
        for j,row in enumerate(matrix):
            print(f"{vr[j]} ",end=" ")
            for k in row:
                print(f"{k}  ",end=" ")
            print()
        print()

    def create_matrix_incidence(self)->None:
        vr = sorted(self.total_graph.keys())
        ln1 = len(vr)
        edge = []
        for i in self.total_graph:
            for j in self.total_graph[i]:
                if (j, i) not in edge and (i, j) not in edge:
                    edge.append((i, j))

        num_edges = len(edge)
        matrix = [[0] * num_edges for _ in range(ln1)]

        dict_vertex = {vertex: i for i, vertex in enumerate(vr)}
        for k, (i, j) in enumerate(edge):
            matrix[dict_vertex[i]][k] = 1
            matrix[dict_vertex[j]][k] = 1
        self.print_matrix_incidence(matrix, vr, edge)




    def print_matrix_incidence(self,matrix:list,vr:list,edge:list)->None:
        print("Таблица инцидентности")
        ln1 = len(matrix)
        ln2 = len(edge)
        for i in range(len(edge)):
            print(f"   {edge[i][0]}-{edge[i][1]}",end=" ")
        print()
        for i in range(ln1):
            print(f"{vr[i]} ", end="")
            for j in range(ln2):
                print(f"  {matrix[i][j]}    ", end="")
            print()
        print()
    def list_of_adjacency(self):
        ln = len(self.total_graph) + 1
        num = [i for i in range(1,ln)]
        d1 = []
        for i in self.total_graph:
            d1.append(i)
        d2 = []
        for j in self.total_graph:
            d2.append(self.total_graph[j])
        data = {'Номер' : num,
                'Вершина' : d1,
                'Смежные вершины' : d2}
        df = pd.DataFrame(data)
        print(tabulate(df,headers="keys",tablefmt="grid",showindex=False))

    #алгоритмы
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
    def degrees_for_each_vert(self)->dict:
        d = {}
        for i in self.total_graph:
            d[i] = len(self.total_graph[i])
        return d
    def BFS(self,start_node:str):
        visited = set()
        queue = Queue()
        queue.enqueue(start_node)


        while not queue.is_empty():
            vertex = queue.dequeue()
            if vertex not in visited:
                visited.add(vertex)
                print(visited)
                for neighbor in self.total_graph.get(vertex, []):
                    if neighbor not in visited:
                        print(neighbor)
                        queue.enqueue(neighbor)
        return queue

    def depth(self,vertex_beg,vertex_end):
        visited = set()
        paths = []
        def dfs(node,current:list):
            visited.add(node)
            current.append(node)
            if node == vertex_end:
                paths.append(current.copy())
            else:
                for i in self.total_graph.get(node,[]):
                    if i not in visited:
                        dfs(i,current)
            visited.remove(node)
            current.pop()
        dfs(vertex_beg,[])
        print('Поиск в глубину')
        print(self.print_depth(paths,paths[0][0],paths[0][-1]))
    def print_depth(self,path,ver1,ver_end):

        data = {'1 вершина': ver1,
                'Последня вершина': ver_end,
                'путь': path}
        df = pd.DataFrame(data)

        print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))





    def has_gamilton(self):
        def dfs(u, path, visited):
            visited.add(u)
            path.append(u)

            if len(path) == len(self.total_graph):
                if any(nei == path[0] for nei in self.total_graph.get(path[-1], [])):
                    return path

            for v in self.total_graph.get(u, []):
                if v not in visited:
                    result = dfs(v, path.copy(), visited.copy())
                    if result:
                        return result

            return None

        for start_node in self.total_graph:
            result = dfs(start_node, [], set())
            if result:
                return result
        return None

    def is_count_vertex_even(self):
        for node in self.total_graph:
            if len(self.total_graph.get(node, [])) % 2 != 0:
                return False
        return True
    def way_1_to_draw(self):
        s = dict(sorted(self.total_graph.items(), key=lambda i: len(i[1]), reverse=True))
        list_keys = list(s.keys())
        tot = [[] for _ in range(3)]
        vis = set()

        for group_index in range(3):
            if not list_keys:
                break

            first_vertex = list_keys[0]
            vis.add(first_vertex)
            tot[group_index].append(first_vertex)
            list_keys.pop(0)

            for vertex in list(list_keys):
                is_non_adjacent_to_all = True

                for existing_vertex in tot[group_index]:
                    if vertex not in self.not_adj_ver(existing_vertex):
                        is_non_adjacent_to_all = False
                        break

                if is_non_adjacent_to_all and vertex not in vis:
                    vis.add(vertex)
                    tot[group_index].append(vertex)
                    list_keys.remove(vertex)

        return tot
    def visial_way_1_to_draw(self):
        total = self.way_1_to_draw()
        num = [i for i in range(1,len(total)+1)]
        color = [total[j] for j in range(len(total))]
        data = {'Цвет': num,'Вершины': color}
        df = pd.DataFrame(data)

        print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))












    def way_2_to_draw(self):
        s = dict(sorted(self.total_graph.items(), key=lambda i: len(i[1]), reverse=True))
        list_keys = list(s.keys())
        tot = [[] for _ in range(3)]
        vis = set()

        for group_index in range(3):
            if not list_keys:
                break

            first_vertex = list_keys[0]
            vis.add(first_vertex)
            tot[group_index].append(first_vertex)
            list_keys.pop(0)

            for vertex in list(list_keys):
                is_non_adjacent_to_all = True

                for existing_vertex in tot[group_index]:
                    if vertex not in self.not_adj_ver(existing_vertex):
                        is_non_adjacent_to_all = False
                        break

                if is_non_adjacent_to_all and vertex not in vis:
                    vis.add(vertex)
                    tot[group_index].append(vertex)
                    list_keys.remove(vertex)

        return tot
    def visial_way_2_to_draw(self):
        total = self.way_2_to_draw()
        num = [i for i in range(1,len(total)+1)]
        color = [total[j] for j in range(len(total))]
        data = {'Цвет': num,'Вершины': color}
        df = pd.DataFrame(data)

        print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))















gr = Graph()
gr.edge_to_add('1', '2')
gr.edge_to_add('1', '3')
gr.edge_to_add('1', '5')
gr.edge_to_add('2', '4')
gr.edge_to_add('2', '3')
gr.edge_to_add('3', '4')
gr.edge_to_add('3', '6')
gr.edge_to_add('3', '5')
gr.edge_to_add('4', '5')
gr.edge_to_add('4', '7')
gr.edge_to_add('5', '8')
gr.edge_to_add('5', '9')
gr.edge_to_add('6', '9')
gr.edge_to_add('5', '6')
gr.edge_to_add('5', '7')
gr.edge_to_add('7', '10')
gr.edge_to_add('7', '8')
gr.edge_to_add('8', '9')
gr.edge_to_add('8', '10')
gr.edge_to_add('9', '10')















