"""
Install Python wrapper with 'pip install metis'
METIS can be downloaded here: http://glaros.dtc.umn.edu/gkhome/views/metis
For further help check https://metis.readthedocs.io/en/latest/
"""

from metis import part_graph
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

