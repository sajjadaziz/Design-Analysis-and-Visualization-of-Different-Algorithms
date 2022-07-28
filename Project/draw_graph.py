import eel
import math
import networkx as nx
import matplotlib.pyplot as plt

def read_file(inputfile):
    graph = []
    vertices = []
    x = []
    y = []
    edges = []
    n = 0

    f = open(inputfile,'r')

    i = 0
    for line in f:
        if(i==2):
            n = int(line)
            break
        i+=1

    i = 0
    for line in f:
        if(i>n):
            break
        for word in line.split():
            graph.append(word)
        for j in range(0, len(graph), 3):
            vertices.append(int(graph[j]))
            x.append(float(graph[j+1]))
            y.append(float(graph[j+2]))
        graph.clear()
        i+=1

    for line in f:
        i = 0
        for word in line.split():
            if(i==0):
                v = int(word)
                i = 1
                continue
            graph.append(word)
        for i in range(0, len(graph), 4):
            edges.append((v, int(graph[i]), float(graph[i+2])))
        graph.clear()
    return n, x, y, vertices, edges

def draw_graph(n, x, y, vertices, edges):
    G = nx.DiGraph()
    for i in range(0, n):
        G.add_node(vertices[i],pos=(x[i],y[i]))
    G.add_weighted_edges_from(edges)
    pos=nx.get_node_attributes(G,'pos')
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edge_labels(G, pos)
    path = "front-end/graph.jpg"
    image = "graph.jpg"
    plt.savefig(path)
    eel.show_graph(image)
    plt.show()

def min_distance(n, dist, s_path):
    min = math.inf
    for i in range(0, n):
        if dist[i] < min and s_path[i] == 0:
            min = dist[i]
            min_index = i
    return min_index

def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, a, b):
    xx = find(parent, a)
    yy = find(parent, b)
    if rank[xx] < rank[yy]:
        parent[xx] = yy
    elif rank[xx] > rank[yy]:
        parent[yy] = xx
    else:
        parent[yy] = xx
        rank[xx] += 1

@eel.expose
def collect_data(inputfile):
    n, x, y, vertices, edges = read_file(inputfile)
    draw_graph(n, x, y, vertices, edges)

@eel.expose
def prim(inputfile):
    n, x, y, vertices, edges = read_file(inputfile)
    adjacency_matrix = [[math.inf for i in range(n)] for j in range(n)]
    p_edges = []
    cost = 0
    for i in range(0, len(edges)):
        xx = edges[i][0]
        yy = edges[i][1]
        ww = edges[i][2]
        if ww < adjacency_matrix[xx][yy]:
            adjacency_matrix[xx][yy] = ww
            adjacency_matrix[yy][xx] = ww
    for i in range(0, n):
        for j in range(0, n):
            if(adjacency_matrix[i][j]==math.inf):
                adjacency_matrix[i][j] = 0.0
    selected = [0]*n
    no_edge = 0
    selected[0] = True
    while (no_edge < n - 1):
        minimum = math.inf
        a = 0
        b = 0
        for i in range(n):
            if selected[i]:
                for j in range(n):
                    if ((not selected[j]) and adjacency_matrix[i][j]):
                        if minimum > adjacency_matrix[i][j]:
                            minimum = adjacency_matrix[i][j]
                            a = i
                            b = j
        p_edges.append((a, b, adjacency_matrix[a][b]))
        cost += adjacency_matrix[a][b]
        selected[b] = True
        no_edge += 1
    
    G = nx.Graph()
    for i in range(0, n):
        G.add_node(vertices[i],pos=(x[i],y[i]))
    G.add_weighted_edges_from(p_edges)
    pos=nx.get_node_attributes(G,'pos')
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edge_labels(G, pos)
    plt.savefig("front-end/prim.jpg")
    eel.show_prim_kruskal_boruvka("prim.jpg", cost)
    plt.show()

@eel.expose
def kruskal(inputfile):
    n, x, y, vertices, edges = read_file(inputfile)
    k_edges = []
    k = 0
    edges = sorted(edges, key=lambda item: item[2])
    parent = []
    rank = []
    for i in range(n):
        parent.append(i)
        rank.append(0)
    i = 0
    while k < n-1:
        xx = edges[i][0]
        yy = edges[i][1]
        ww = edges[i][2]
        i+=1
        a = find(parent, xx)
        b = find(parent, yy)
        if a != b:
            k+=1
            k_edges.append((xx, yy, ww))
            union(parent, rank, a, b)

    cost=0
    for i in range(0, len(k_edges)):
        cost+=k_edges[i][2]
    G = nx.Graph()
    for i in range(0, n):
        G.add_node(vertices[i],pos=(x[i],y[i]))
    G.add_weighted_edges_from(k_edges)
    pos=nx.get_node_attributes(G,'pos')
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edge_labels(G, pos)
    plt.savefig("front-end/kruskal.jpg")
    eel.show_prim_kruskal_boruvka("kruskal.jpg", cost)
    plt.show()

@eel.expose
def dijkstra(inputfile):
    n, x, y, vertices, edges = read_file(inputfile)
    src = 0
    dist = [math.inf]*n
    dist[src] = 0
    s_path = [0]*n
    adjacency_matrix = [[math.inf for i in range(n)] for j in range(n)]
    for i in range(0, len(edges)):
        xx = edges[i][0]
        yy = edges[i][1]
        ww = edges[i][2]
        if ww < adjacency_matrix[xx][yy]:
            adjacency_matrix[xx][yy] = ww
            adjacency_matrix[yy][xx] = ww
    for i in range(0, n):
        for j in range(0, n):
            if(adjacency_matrix[i][j]==math.inf):
                adjacency_matrix[i][j] = 0.0
    for i in range(0, n):
        xx = min_distance(n, dist, s_path)
        s_path[xx] = True
        for yy in range(0, n):
            if adjacency_matrix[xx][yy] > 0 and s_path[yy] == 0 and dist[yy] > dist[xx] + adjacency_matrix[xx][yy]:
                dist[yy] = dist[xx] + adjacency_matrix[xx][yy]
    print("\nDijkstra Algorithm\nSource     Destination     Cost")
    for i in range(0, len(vertices)):
        print(edges[0][0], "             ", vertices[i], "        ", dist[i])
    eel.show_dij_bell(edges[0][0], vertices, dist)

@eel.expose
def bellman(inputfile):
    n, x, y, vertices, edges = read_file(inputfile)
    dist = [math.inf]*n
    edges = sorted(edges, key=lambda item: item[0])
    src = edges[0][0]
    dist[src] = 0
    for i in range(0, n-1):
        for j in range(0, len(edges)):
            xx = edges[j][0]
            yy = edges[j][1]
            ww = edges[j][2]
            if dist[xx] != math.inf and dist[yy] > dist[xx] + ww:
                dist[yy] = dist[xx] + ww
 
    for i in range(len(edges)):
        xx = edges[i][0]
        yy = edges[i][1]
        ww = edges[i][2]
        if dist[xx] != math.inf and dist[yy] > dist[xx] + ww:
            print("Graph contains negative weight cycle")
    print("\nBellman Ford Algorithm\nSource     Destination     Cost")
    for i in range(0, len(vertices)):
        print(edges[0][0], "             ", vertices[i], "        ", dist[i])
    eel.show_dij_bell(edges[0][0], vertices, dist)

@eel.expose
def warshall(inputfile):
    n, x, y, vertices, edges = read_file(inputfile)
    adjacency_matrix = [[math.inf for i in range(n)] for j in range(n)]
    dist = [[math.inf for i in range(n)] for j in range(n)]
    for i in range(0, len(edges)):
        xx = edges[i][0]
        yy = edges[i][1]
        ww = edges[i][2]
        if ww < adjacency_matrix[xx][yy]:
            adjacency_matrix[xx][yy] = ww
            adjacency_matrix[yy][xx] = ww
            dist[xx][yy] = ww
            dist[yy][xx] = ww
    for i in range(0, n):
        for j in range(0, n):
            if(i==j):
                adjacency_matrix[i][j] = 0.0
                dist[i][j] = 0.0
    
    for k in range(0, n):
        for i in range(0, n):
            for j in range(0, n):
                dist[i][j] = min(dist[i][j], dist[i][k]+dist[k][j])
    print("\nFloyd Warshall Algorithm")
    for i in range(0, n):
        print(dist[i])
    eel.show_warshall(dist, n)

@eel.expose
def clustering(inputfile):
    n, x, y, vertices, edges = read_file(inputfile)
    c_edges = []
    coefficient = [0]*n
    for i in range(0, len(edges)):
        c_edges.append((edges[i][0], edges[i][1]))
    G = nx.Graph()
    G.add_edges_from(c_edges)
    for i in range(0, n):
        coefficient[i] = nx.clustering(G, i)
    print("\nClustering Coefficient Algorithm\nNode     Coefficient")
    for i in range(0, n):
        print(i, "        ", coefficient[i])
    eel.show_clustering(coefficient, n)

@eel.expose
def boruvka(inputfile):
    n, x, y, vertices, edges = read_file(inputfile)
    b_edges = []
    parent = []
    rank = []
    min = []
    cost = 0
    for i in range(n):
        parent.append(vertices[i])
        rank.append(0)
        min = [-1]*n
    i = n
    while i > 1:
        for j in range(0, len(edges)):
            xx = edges[j][0]
            yy = edges[j][1]
            ww = edges[j][2]
            a = find(parent, xx)
            b = find(parent ,yy)
            if a != b:
                if min[a] == -1 or min[a][2] > ww :
                    min[a] = [xx, yy, ww]
                if min[b] == -1 or min[b][2] > ww :
                    min[b] = [xx, yy, ww]
        for j in range(n):
            if min[j] != -1:
                xx = min[j][0]
                yy = min[j][1]
                ww = min[j][2]
                a = find(parent, xx)
                b = find(parent ,yy)
                if a != b :
                    cost += ww
                    union(parent, rank, a, b)
                    b_edges.append((xx, yy, ww))
                    i-=1
        min=[-1]*n
    G = nx.Graph()
    for i in range(0, n):
        G.add_node(vertices[i],pos=(x[i],y[i]))
    G.add_weighted_edges_from(b_edges)
    pos=nx.get_node_attributes(G,'pos')
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edge_labels(G, pos)
    plt.savefig("front-end/boruvka.jpg")
    eel.show_prim_kruskal_boruvka("boruvka.jpg", cost)
    plt.show()

eel.init("front-end")
eel.start("index.html")