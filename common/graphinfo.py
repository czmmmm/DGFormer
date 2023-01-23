import numpy as np
import torch
import networkx as nx
import scipy.sparse as sp


class Graph:
    def __init__(self,  dataset, kp=None):
        if dataset == 'h36m':
            if kp == 'sh_pt_mpii':
                self.num_nodes = 16
                neighbor_base = [(6, 2), (2, 1), (1, 0), (6, 3), (3, 4), (4, 5), (6, 7),
                                 (7, 8), (8, 9), (8, 12), (12, 11),
                                 (11, 10), (8, 13), (13, 14), (14, 15)]
            else:
                self.num_nodes = 17
                neighbor_base = [(0, 1), (2, 1), (3, 2), (4, 0), (5, 4), (6, 5),
                                 (7, 0), (8, 7), (9, 8), (10, 9), (11, 8),
                                 (12, 11), (13, 12), (14, 8), (15, 14), (16, 15)]
        elif dataset == 'mpi3dhp':
            self.num_nodes = 17
            neighbor_base = [(0, 1), (2, 1), (3, 2), (4, 0), (5, 4), (6, 5),
                             (7, 0), (8, 7), (9, 8), (10, 9), (11, 8),
                             (12, 11), (13, 12), (14, 8), (15, 14), (16, 15)]
        elif dataset == 'humaneva15':
            self.num_nodes = 15
            neighbor_base = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                             (0, 8), (8, 9), (9, 10), (0, 11), (11, 12), (12, 13), (0, 14)]
        self.edges = neighbor_base
        G = nx.Graph()
        G.add_edges_from(self.edges)
        self.degrees = G.degree
        self.adjaceny = nx.to_numpy_array(G)
        self.ai = self.adjaceny + np.eye(self.num_nodes)
        self.adj1 = torch.from_numpy(self.adjaceny.astype('float32'))
        self.normadj1 = torch.from_numpy(normalize_undigraph(self.ai).astype('float32'))

    @staticmethod
    def normalization(adjacency):
        """L=D^--.5 * (A + I) * D^-0.5"""
        adjacency += sp.eye(adjacency.shape[0])
        degree = np.array(adjacency.sum(1))
        d_hat = sp.diags(np.power(degree, -0.5).flatten())
        return d_hat.dot(adjacency).dot(d_hat)

    @staticmethod
    def k_adjacency(A, k, with_self=False, self_factor=1):
        assert isinstance(A, np.ndarray)
        I = np.eye(len(A), dtype=A.dtype)
        if k == 0:
            return I
        Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
             - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
        if with_self:
            Ak += (self_factor * I)
        return Ak
    @staticmethod
    def normalize_adjacency_matrix(A):
        node_degrees = A.sum(-1)
        degs_inv_sqrt = np.power(node_degrees, -0.5)
        norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
        return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)



def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

if __name__ == '__main__':
    graph = Graph('h36m')
    adj = graph.adjaceny
    normadj = graph.normalization(adj)


    print('helllo')