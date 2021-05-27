import networkx as nx


def build_memetracker_graph(pathtofile, graphtype):
    node_dict = {}

    if graphtype == 'undirected':
        g = nx.Graph()
    elif graphtype == 'directed':
        g = nx.DiGraph()
    else:
        print('Unrecognized graph type .. aborting!')
        return -1

    times = []
    with open(pathtofile) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = content[1:]

    for line in content:
        entries = line.split()
        src_str = entries[1]
        dst_str = entries[2]

        if src_str not in node_dict:
            node_dict[src_str] = len(node_dict)
            g.add_node(node_dict[src_str])
        if dst_str not in node_dict:
            node_dict[dst_str] = len(node_dict)
            g.add_node(node_dict[dst_str])

        src_idx = node_dict[src_str]
        dst_idx = node_dict[dst_str]

        g.add_edge(src_idx, dst_idx)

        times.append(float(entries[-1]))

    return g, node_dict