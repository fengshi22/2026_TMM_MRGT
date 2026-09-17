import numpy as np
import torch
import scipy.io
import hdf5storage
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

def load_relation_knn(multi_view_features, k):
    adj_load = []
    for idx, features in enumerate(multi_view_features[0]):
        adj = construct_adjacency_matrix(features, k_nearest_neighbors=k)
        adj_load.append(adj.todense())
    adj_list = torch.from_numpy(np.array(adj_load)).float()
    edge_indices = [torch.nonzero(adj, as_tuple=False).cuda() for adj in adj_list]
    adj_list = [adj.cuda() for adj in adj_list]
    return edge_indices, adj_list


def construct_adjacency_matrix(
        features,
        k_nearest_neighbors,
        prunning_one=False,
        prunning_two=True,
        common_neighbors=2
):
    nbrs = NearestNeighbors(n_neighbors=k_nearest_neighbors + 1, algorithm='ball_tree').fit(features)
    adj_wave = nbrs.kneighbors_graph(features)

    if prunning_one:
        original_adj_wave = adj_wave.toarray()
        judges_matrix = original_adj_wave == original_adj_wave.T
        np_adj_wave = original_adj_wave * judges_matrix
        adj_wave = sp.csc_matrix(np_adj_wave)
    else:
        np_adj_wave = construct_symmetric_matrix(adj_wave.toarray())
        adj_wave = sp.csc_matrix(np_adj_wave)

    adj = sp.csc_matrix(np_adj_wave)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    if prunning_two:
        adj = adj.toarray()
        b = np.nonzero(adj)
        rows = b[0]
        cols = b[1]
        dic = {}
        for row, col in zip(rows, cols):
            if row in dic.keys():
                dic[row].append(col)
            else:
                dic[row] = []
                dic[row].append(col)
        for row, col in zip(rows, cols):
            if len(set(dic[row]) & set(dic[col])) < common_neighbors:
                adj[row][col] = 0
        adj = sp.csc_matrix(adj)
        adj.eliminate_zeros()
    return adj


def construct_symmetric_matrix(original_matrix):
    result_matrix = np.zeros(original_matrix.shape, dtype=float)
    num = original_matrix.shape[0]
    for i in range(num):
        for j in range(num):
            if original_matrix[i][j] == 0:
                continue
            elif original_matrix[i][j] == 1:
                result_matrix[i][j] = 1
                result_matrix[j][i] = 1
            else:
                print("The value in the original matrix is illegal!")
    assert (result_matrix == result_matrix.T).all() == True

    if ~(np.sum(result_matrix, axis=1) > 1).all():
        print("There existing a outlier!")

    return result_matrix


