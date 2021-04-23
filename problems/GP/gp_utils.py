import networkx as nx
from metis import part_graph
import matplotlib.pyplot as plt
from networkx.generators import barabasi_albert_graph
import numpy as np
import random


def get_metis_partition(graph, recurisve=False, nparts=2):
    """
    :param graph: A NetworkX graph
    :param recurisve: Determines whether the partitioning should be done by direct k-way cuts or
     by a series of recursive cuts. These correspond to METIS_PartGraphKway() and
     METIS_PartGraphRecursive() in METIS’s C API.
    :return: number of edges between partitions
    """
    cost, parts = part_graph(graph=graph, nparts=nparts, recursive=recurisve, tpwgts=[.5, .5], ufactor=1)
    partition_nodes = np.sum(parts)
    # print('partition_nodes: ', parts)
    if partition_nodes != graph.number_of_nodes() / 2:
        # partition is not balanced
        return -1
    else:
        return cost


def random_partition(graph):
    """
    :param graph: A NetworkX graph
    :return: number of edges between node sets partitions. The partition is generated from a random
    ordering of vertices. Each vertex is added to set such that it increases the number of edges
    in the cut the least. If on partition is full, the remaining nodes are added to the other one.
    """
    n_nodes = graph.number_of_nodes()
    permutation = np.random.permutation(n_nodes)
    p1_size = 0
    p2_size = 0
    cut_size = 0

    partition = [-1] * n_nodes
    # count for each node the neighbors it has in each of the two node sets.
    partition_neighbor_count = [[0, 0] for _ in range(n_nodes)]

    def update_routine(current_node, assigned_partition):

        s1, s2 = partition_neighbor_count[current_node]

        nonlocal p1_size, p2_size, cut_size

        if assigned_partition == 0:
            p1_size += 1
        else:
            p2_size += 1

        for u in graph.neighbors(current_node):
            partition_neighbor_count[u][assigned_partition] += 1

        cut_size = cut_size + s1 if assigned_partition == 1 else cut_size + s2

    for i, node in enumerate(permutation):
        s1, s2 = partition_neighbor_count[node]
        if p1_size >= (n_nodes / 2) or p2_size >= (n_nodes / 2) or s1 == s2:
            if p1_size > p2_size:
                # node is added to p2
                update_routine(node, 1)
                partition[node] = 1
            elif p1_size < p2_size:
                # node is added to P1
                update_routine(node, 0)
                partition[node] = 0
            else:
                rand_bit = random.getrandbits(1)
                if rand_bit:
                    update_routine(node, 0)
                    partition[node] = 0
                else:
                    update_routine(node, 1)
                    partition[node] = 1
        elif s1 > s2:
            # node has more neighbors in P1 -> add this node to P1
            update_routine(node, 0)
            partition[node] = 0
        else:
            assert s1 < s2
            update_routine(node, 1)
            partition[node] = 1

    return cut_size, partition
