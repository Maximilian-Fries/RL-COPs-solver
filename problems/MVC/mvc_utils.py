import torch_geometric.utils as torch_utils
import torch_geometric
from networkx.algorithms.approximation import min_weighted_vertex_cover
import operator
import numpy as np
import cplex
import networkx


def list_right(graph):
    if isinstance(graph, torch_geometric.data.Data):
        graph = torch_utils.to_networkx(graph, to_undirected=True)
    degrees = [val for (node, val) in graph.degree()]
    inds = list(range(len(degrees)))
    inds_sorted = sorted(inds, key=lambda ind: degrees[ind])
    C = set()
    for node in inds_sorted[::-1]:
        for u in graph.neighbors(node):
            if u not in C:
                C.add(node)
                break
    return C


# extremely inefficient, but does the job
def get_iter_greedy(graph):
    if isinstance(graph, torch_geometric.data.Data):
        graph = torch_utils.to_networkx(graph, to_undirected=True)
    graph = graph.copy()
    edge_deg = dict.fromkeys(graph.edges)
    vc = set()
    m = len(graph.edges)
    for edge in graph.edges:
        edge_deg[edge] = graph.degree(edge[0]) + graph.degree(edge[1])
    while m>0:
        max_edge = max(edge_deg.items(), key=operator.itemgetter(1))[0]
        vc.update(max_edge)
        # remove edges from graph
        to_remove_edges = set(graph.edges(max_edge))
        graph.remove_edges_from(to_remove_edges)

        m = m - len(to_remove_edges)

        for e in to_remove_edges:
            del edge_deg[(min(e[0], e[1]), max(e[0], e[1]))]

        for edge in graph.edges:
            edge_deg[edge] = graph.degree(edge[0]) + graph.degree(edge[1])
    return vc


def random_cover(graph):
    if isinstance(graph, networkx.classes.graph.Graph):
        graph = torch_utils.from_networkx(graph)
    n = graph.num_nodes
    pi = np.random.permutation(n)
    subset = []
    edge_index = graph.edge_index
    mask = np.zeros(n, dtype=np.bool)
    for v in pi:
        mask[subset] = 1
        covered_edges = mask[edge_index[0]] | mask[edge_index[1]]
        if np.all(covered_edges):
            return len(subset)
        subset.append(v.item())


def get_heuristic_cover(graph, strategy):

    if strategy == 'iterative':
        if isinstance(graph, torch_geometric.data.Data):
            graph = torch_utils.to_networkx(graph, to_undirected=True)
        return len(min_weighted_vertex_cover(graph))

    elif strategy == 'iter_greedy':
        return len(get_iter_greedy(graph))
    elif strategy == 'list_right':
        return len(list_right(graph))
    elif strategy == 'random':
        return random_cover(graph)


def cplex_solve(graph):
    prob = cplex.Cplex()
    '''prob.set_log_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)'''
    prob.parameters.timelimit.set(100.0)
    prob.set_problem_name("Minimum Vertex Cover")
    prob.set_problem_type(cplex.Cplex.problem_type.LP)

    prob.objective.set_sense(prob.objective.sense.minimize)
    names = list(map(str,list(graph.nodes)))
    n = graph.number_of_nodes()
    w_obj = [1]*n
    low_bnd = [0]*n
    upr_bnd = [1]*n
    prob.variables.add(names=names, obj=w_obj, lb=low_bnd, ub=upr_bnd)
    all_int = [(var, prob.variables.type.integer) for var in names]
    prob.variables.set_types(all_int)

    edges = [[str(x), str(y)] for x,y in list(graph.edges)]
    constraints = list(zip(edges, [[1,1] for _ in range(len(edges))]))

    constraint_names = ["".join(x[0]) for x in constraints]

    # Each edge must have at least one vertex
    rhs = [1] * len(constraints)

    constraint_senses = ["G"] * len(constraints)

    prob.linear_constraints.add(names=constraint_names,
                                lin_expr=constraints,
                                senses=constraint_senses,
                                rhs=rhs)

    #print("Problem Type: %s" % prob.problem_type[prob.get_problem_type()])
    prob.solve()
    #print("Solution result is: %s" % prob.solution.get_status_string())
    return np.sum(prob.solution.get_values())
