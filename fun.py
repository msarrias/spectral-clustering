import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import networkx as nx
from math import sqrt, exp
import scipy.linalg as la
from sklearn import cluster, datasets
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph


def read_and_convert_data(data_path, labels_path):
    """
    read_and_convert_data_points is a function that takes
    a text file as an argument and converts it into an 
    array containing all the datapoints.
    ======================================================
    :param filename: text file name in directory.
    :return: Npts x 2 array, labels vector of length Npts. 
    ======================================================
    """
    splits = []
    with open(data_path, 'r') as d, open(labels_path, 'r') as l:
                data_points = d.readlines()
                data_labels = l.readlines()
    
    for data_point in data_points:
        splits.append(data_point.split())
    
    return (np.array(splits, dtype=np.float),
            np.array(data_labels, dtype=np.int))

def scatter_plot_data_set(data,labels, color_clusters = True):
    """
    scatter_plot_data_set is a function that plots the data.
     ======================================================
    :param data: Npts x 2, data set array.
    :labels: Npts labels vector.
    :return:
     ======================================================
    """
    x, y = data.T
    colors = ['red','orange','blue','green']
    if color_clusters:
        plt.scatter(x,y,c=labels, 
                cmap=matplotlib.colors.ListedColormap(colors))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
    else:
        plt.scatter(x,y, c="black")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
        
def eucledian_dist(x_i, x_j):
    """
    eucledian_dist is a function that 
     ======================================================
    :param x_i: 
    :x_j: 
    :return:
     ======================================================
    """
    coord = x_i.shape[0]
    d=[]
    if coord == x_j.shape[0]:
        for i in range(coord):
            d.append((x_i[i] - x_j[i])**2)
    return (np.sqrt(sum(d),dtype=np.float64))

def distance_matrix(data, distance_measure):
    """
     ======================================================
    :param data: 
    :distance_measure: 
    :return:
     ======================================================
    """
    Npts= data.shape[0]
    distance_matrix=np.zeros((Npts,Npts))
    for xi in range(Npts):
        for xj in range(Npts):
            distance_matrix[xi,xj] = distance_measure(data[xi],data[xj])
    return(distance_matrix)

def adjacency_matrix(data, sigma):
    """
    adjacency_matrix is a function that 
     ======================================================
    :param data: 
    :sigma: 
    :return:
     ======================================================
    """
    dist_matrix = distance_matrix(data, eucledian_dist)
    adjacency_matrix= np.exp(-(dist_matrix)**2 /sigma)
    adjacency_matrix[adjacency_matrix==1] = 0
    return(adjacency_matrix)

def diagonal_matrix(adjacency_matrix):
    """
    diagonal_matrix is a function that 
     ======================================================
    :param adjacency_matrix: 
    :return:
     ======================================================
    """
    return(np.diag(sum(adjacency_matrix)))


def unnormalized_graph_Laplacian(adjacency_matrix):
    """
    unnormalized_graph_Laplacian is a function that 
     ======================================================
    :param adjacency_matrix: 
    :return:
     ======================================================
    """
    diag_matrix = diagonal_matrix(adjacency_matrix)
    if diag_matrix.shape == adjacency_matrix.shape:
        return(diag_matrix - adjacency_matrix)
    

def normalized_graph_Laplacian(adjacency_matrix, matrix = "symmetric"):
    """
    normalized_graph_Laplacian is a function that 
     ======================================================
    :param adjacency_matrix: 
    :matrix:
    :return:
     ======================================================
    """
    D = diagonal_matrix(adjacency_matrix)
    L = unnormalized_graph_Laplacian(adjacency_matrix)
    if matrix == "symmetric":
        return(np.matmul(np.matmul(np.diag(sum(D)**(-1/2)),L)
                         ,np.diag(sum(D)**(-1/2))))
    if matrix == "rw":
        return(np.matmul(np.diag(sum(D)**(-1)), L))
    
def create_weighted_Graph(W):
    """
    create_weighted_Graph is a function that 
     ======================================================
    :param W: 
    :return: graph
     ======================================================
    """
    Npts = W.shape[0]
    nodes_idx = [i for i in range(Npts)]
    graph = nx.Graph()
    graph.add_nodes_from(nodes_idx)
    #graph.add_edges_from([(i,j) for i in range(Npts)for j in range(Npts)])
    graph.add_weighted_edges_from([(i,j, W[i][j])
                                   for i in range(Npts) for j in range(Npts)])
    return(graph)  


def plot_Graph(graph, nodes_position, title = '', node_size=20,
               alpha=0.3, edge_vmax=1e-1, output_file_name="none"):
    """
    plot_Graph is a function that 
     ======================================================
    :param graph: 
    :nodes_position:
    :title:
    :alpha:
    :edge_vmax:
    :output_file_name:
    :return: plot
     ======================================================
    """
    if output_file_name=="none":
        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(graph, nodes_position, 
                               node_size=node_size, node_color="red") 
        nx.draw_networkx_edges(graph, nodes_position,
                               edge_cmap= plt.cm.Blues,
                               width=1.5, edge_vmax=edge_vmax, 
                               edge_color=[graph[u][v]['weight'] 
                                           for u, v in graph.edges],
                               alpha=alpha)
        plt.axis('off')
        plt.title(title)
        plt.show()
    else:
        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(graph, nodes_position, 
                               node_size=node_size, node_color="red") 
        nx.draw_networkx_edges(graph, nodes_position,
                               edge_cmap= plt.cm.Blues,
                               width=1.5, edge_vmax=edge_vmax, 
                               edge_color=[graph[u][v]['weight'] 
                                           for u, v in graph.edges],
                               alpha=alpha)
        plt.axis('off')
        plt.title(title)
        plt.savefig(output_file_name)
        plt.show()

def generate_3circles_data_set(Npts_list, rad_list, lower_boundry_list, seed=1991):
    """
    generate_3circles_data_set is a function that 
     ======================================================
    :param Npts_list: 
    :rad_list:
    :lower_boundry_list:
    :seed:
    :return: circle_data_points
     ======================================================
    """
    
    Ncircles = 3
    np.random.seed(seed) 
    circle_data_points = np.zeros((sum(Npts_list),3))

    for i in range(Ncircles):        
        t = np.random.uniform(low=0.0, high=2.0*np.pi,size=Npts_list[i])
        r = rad_list[i] * np.sqrt(np.random.uniform(low=lower_boundry_list[i], 
                                                    high=1, size=Npts_list[i]))
     
        if i==0:
            circle_data_points[i:Npts_list[i], 0] = r * np.cos(t)
            circle_data_points[i:Npts_list[i], 1] = r * np.sin(t)
            circle_data_points[i:Npts_list[i], 2] = [i] * Npts_list[i]
          
        lower = sum(Npts_list[0:i])  
        upper =sum(Npts_list[0:i+1])
        circle_data_points[lower:upper, 0] = r * np.cos(t)
        circle_data_points[lower:upper, 1] = r * np.sin(t)
        circle_data_points[lower:upper, 2] = [i] * Npts_list[i]
        
    return(circle_data_points)
