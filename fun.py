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
    :return: plot
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
    eucledian_dist is a function that computes the eucledian
    distance between two data points.
     ======================================================
    :param x_i, x_j: data points "i" and "j" in a n_coord
                    system.
    :return: Scalar, Eucledian distance of two data points.
     ======================================================
    """
    n_coord = x_i.shape[0]
    d=[]
    if n_coord == x_j.shape[0]:
        for i in range(n_coord):
            d.append((x_i[i] - x_j[i])**2)
    return (np.sqrt(sum(d),dtype=np.float64))

def commute_time_distance(i, j, scale_factor, eigenvectors_matrix, D):
    """
    commute_time_distance is a function that computes the
    CTD between node i and node j.
     ======================================================
    :param i, j: node i and node j index.
    :scale_factor: vector of length Npts - 1, where each
    element corresponds to 1/eigenvector_k for the kth
    eigenvector.
    :eigenvectors_matrix: (Npts-1) * Npts matrix containing
    Npts-1 eigenvectors row-wise.
    :D: Diagonal matrix
    :return: Scalar, CTD of two nodes.
     ======================================================
    """
    return(np.sum( scale_factor * 
        (eigenvectors_matrix[:,j]/D[j][j] 
         - eigenvectors_matrix[:,i]/D[i][i])**2))


def distance_matrix(input_, distance_measure, adjacency_matrix =[]):
    """
    distance_matrix is a function that computes the 
    similarity matrix taken pairwise, between the elements 
    of a dataset.
     ======================================================
    :param input_: i) Npts * 2 matrix containing the data 
                   coordinates.
                   ii) sorted dictionary, key: eigenvalue
                                          value: eigenvector.
    :distance_measure: i) Eucledian
                       ii) CTD
    :adjacency_matrix: only needed when distance_measure = CTD.
    :return: Npts * Npts similarity matrix.
     ======================================================
    """
    if distance_measure == "eucledian_dist":
        Npts= input_.shape[0]
        distance_matrix=np.zeros((Npts,Npts))
        
        for xi in range(Npts):
            for xj in range(xi, Npts):
                distance_matrix[xi,xj] = eucledian_dist(
                                         input_[xi],input_[xj])
                distance_matrix[xj,xi] = distance_matrix[xi,xj]
                
        return(distance_matrix)
    
    if distance_measure == "commute_time_distance":
        Npts= len(input_)
        distance_matrix=np.zeros((Npts,Npts))
        eigenvectors_matrix = np.zeros((Npts-1, Npts))
        eigenvalues_symm_list = []
        #Unpack eigenvalues and eigenvectors in a list/matrix
        for i in range(1, Npts):
            eigenvectors_matrix[i-1] = input_[i][1]
            eigenvalues_symm_list.append(input_[i][0])
        #Compute distance matrix
        D = diagonal_matrix(adjacency_matrix)
        #Scaling factor:
        scale_factor = 1 / np.array(eigenvalues_symm_list)
        for i in range(Npts):
            for j in range(i, Npts):
                c_ij= commute_time_distance(i, j, scale_factor, 
                                            eigenvectors_matrix, D)
                distance_matrix[i][j] = c_ij
                distance_matrix[j][i] = c_ij
                
        return(distance_matrix)

def adjacency_matrix(data, sigma):
    """
    adjacency_matrix is a function that returns the matrix
    representation of a graph with connectivity weights 
    given by the Gaussian similarity function.
     ======================================================
    :param data: Coordinates Npts*2 matrix
    :sigma: parameter, extend of the neighborhood
    :return:
     ======================================================
    """
    dist_matrix = distance_matrix(data, "eucledian_dist")
    adjacency_matrix= np.exp(-(dist_matrix)**2 /sigma)
    adjacency_matrix[adjacency_matrix==1] = 0
    return(adjacency_matrix)

def diagonal_matrix(adjacency_matrix):
    """
    diagonal_matrix is a function that returns the diagonal 
    matrix of a weighted graph.
     ======================================================
    :param adjacency_matrix: Npts*Npts matrix
    :return: Npts*Npts Diagonal matrix
     ======================================================
    """
    return(np.diag(sum(adjacency_matrix)))

def incidence_matrix(labels):
    """
    incidence_matrix is a function that shows the relationship
    between two elements in a dataset. E.g if element i and j
    belong to the same cluster, then the incidence will be set
    to 1.
     ======================================================
    :param labels: vector of length Npts.
    :return: Npts*Npts incidence matrix
     ======================================================
    """
    Npts = len(labels)
    incidence_matrix = np.zeros((Npts,Npts))
    for i in range(Npts):
        for j in range(Npts):
            if labels[i] == labels[j]:
                incidence_matrix[i][j] = 1
            else:
                incidence_matrix[i][j] = 0
    return(incidence_matrix)

def unnormalized_graph_Laplacian(adjacency_matrix):
    """
    unnormalized_graph_Laplacian is a function that returns
    the unnormalized graph Laplacian. This matrix will gives
    an idea of how connected each data point is to the rest
    of the network.
     ======================================================
    :param adjacency_matrix: Npts*Npts matrix
    :return: Symmetric, positive, semi-definite
             Npts*Npts matrix
     ======================================================
    """
    diag_matrix = diagonal_matrix(adjacency_matrix)
    if diag_matrix.shape == adjacency_matrix.shape:
        return(diag_matrix - adjacency_matrix)
    

def normalized_graph_Laplacian(adjacency_matrix, 
                               matrix = "symmetric"):
    """
    normalized_graph_Laplacian is a function that computes:
    i) Symmetric Normalized graph Laplacian
    ii)  Random walk normalized graph Laplacian
     ======================================================
    :param adjacency_matrix: Npts*Npts matrix
    :matrix: "symmetric" or "rw"
    :return: Npts*Npts matrix
     ======================================================
    """
    D = diagonal_matrix(adjacency_matrix)
    L = unnormalized_graph_Laplacian(adjacency_matrix)
    if matrix == "symmetric":
        return(np.matmul(np.matmul(np.diag(sum(D)**(-1/2)),L)
                         ,np.diag(sum(D)**(-1/2))))
    if matrix == "rw":
        return(np.matmul(np.diag(sum(D)**(-1)), L))

def correlation_coefficient(M1, M2):
    """
    correlation_coefficient is a function that returns
    the Pearson correlation coefficient between two matrices.
     ======================================================
    :param M1,M2: Npts*Npts matrices.
    :return: scalar, correlation coefficient
     ======================================================
    """
    numerator = np.mean((M1 - M1.mean()) * (M2 - M2.mean()))
    denominator = M1.std() * M2.std()
    return (numerator / denominator)

def create_weighted_Graph(adjacency_matrix):
    """
    create_weighted_Graph is a function that generates a 
    graph from the  by using the `networkx` library in python.
     ======================================================
    :param W: Npts*Npts adjacency matrix
    :return: graph
     ======================================================
    """
    Npts = adjacency_matrix.shape[0]
    nodes_idx = [i for i in range(Npts)]
    graph = nx.Graph()
    graph.add_nodes_from(nodes_idx)
    graph.add_weighted_edges_from([(i,j, adjacency_matrix[i][j])
                                   for i in range(Npts) 
                                   for j in range(Npts)])
    return(graph)  

def plot_Graph(graph, nodes_position, title = '', node_size=20,
               alpha=0.3, edge_vmax=1e-1, output_file_name="none"):
    """
    plot_Graph is a function that draws the input graph.
     ======================================================
    :param graph: Graph object, generated by using the
    `networkx` library.
    :nodes_position: Npts*2 data coordinates
    :title:
    :alpha: (float) – The edge transparency
    :edge_vmax: (float) - maximum for edge colormap scaling
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

def generate_3circles_data_set(Npts_list, rad_list, 
                               lower_boundry_list,
                               seed=1991):
    """
    generate_3circles_data_set is a function that generates
    a data set with three contained size increasing circles.
     ======================================================
    :param Npts_list: List containing the number of data 
    points per circle.
    :rad_list: List containing the radius per circle.
    :lower_boundry_list: Lower boundary of the output interval.
    All values generated will be greater than or equal to 
    lower_boundry_list.
    :seed: for reproducible results
    :return: Npts * 2 matrix
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

