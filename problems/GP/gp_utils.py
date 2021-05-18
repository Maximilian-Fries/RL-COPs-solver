import networkx as nx
from metis import part_graph
import matplotlib.pyplot as plt
from networkx.generators import barabasi_albert_graph
import numpy as np
import random
import cplex


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
        return np.NaN
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


def cplex_solve(graph):
    prob = cplex.Cplex()
    '''prob.set_log_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)'''
    prob.parameters.timelimit.set(100.0)
    prob.set_problem_name("Graph Paritioning")
    prob.set_problem_type(cplex.Cplex.problem_type.LP)

    prob.objective.set_sense(prob.objective.sense.minimize)

    edges = [[str(x), str(y)] for x, y in list(graph.edges)]

    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    names = ['y_{0}_{1}'.format(i, j) for i in range(m) for j in range(m)]  # list(map(str,list(graph.edges)))
    w_obj = [1]*m
    low_bnd = [0]*m
    upr_bnd = [1]*m
    prob.variables.add(names=names, obj=w_obj, lb=low_bnd, ub=upr_bnd)

    nodes_names = ['x' + str(v) for v in range(n)]
    prob.variables.add(names=nodes_names, obj=[1]*n, lb=[0]*n, ub=[1]*n)

    all_int = [(var, prob.variables.type.integer) for var in names]
    prob.variables.set_types(all_int)

    constraints = list(zip(edges, [[1, 1] for _ in range(len(edges))]))
    constraint_names = ["".join(x[0]) for x in constraints]

    # Each edge must have at least one vertex
    rhs = [1] * len(constraints)

    constraint_senses = ["G"] * len(constraints)
    prob.linear_constraints.add(names=constraint_names,
                                lin_expr=constraints,
                                senses=constraint_senses,
                                rhs=rhs)

    prob.linear_constraints.add(names=constraint_names,
                                lin_expr=constraints,
                                senses=constraint_senses,
                                rhs=rhs)

    #print("Problem Type: %s" % prob.problem_type[prob.get_problem_type()])
    prob.solve()
    #print("Solution result is: %s" % prob.solution.get_status_string())
    return np.sum(prob.solution.get_values())