import os, sys
import networkx as nx

assert len(sys.argv) == 3, 'Too few arguments given!'

N = int(sys.argv[1])
k = int(sys.argv[2])

G = nx.random_regular_graph(k, N)
G = nx.relabel_nodes(G, dict(zip(G.nodes, range(N))))

nx.write_edgelist(G, f'edgelist_N{N}.txt', data=False)