import networkx as nx
from metis import part_graph
import matplotlib.pyplot as plt
from networkx.generators import barabasi_albert_graph
import numpy as np


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
    if np.abs(partition_nodes - (graph.number_of_nodes()//2)) > 1:
        # partition is not balanced
        return -1
    else:
        return cost

'''size=500
graph = barabasi_albert_graph(n=size, m=3, seed=12121212)

cost, parts = part_graph(graph=graph, nparts=2, recursive=False, tpwgts=[.5, .5], ufactor=1)
print('cost: {}\nparts: {}'.format(cost, parts))
print('#nodes in partition: ', np.sum(parts))
colors = ['yellow', 'lightblue']
pos = nx.spring_layout(graph)
color_vec = [0]*size
for i, p in enumerate(parts):
    color_vec[i] = colors[p]
    pos[i][0] = pos[i][0] + 2 if parts[i] else pos[i][0] - 2'''




#nx.draw(graph, with_labels=True, node_color=color_vec, pos=pos)
#plt.show()