import os, sys
import numpy as np
import networkx as nx

path = os.path.abspath('../../')
sys.path.append(path)

print(f'Looking for QubitRBM in {path}')

from qubitrbm.qaoa import QAOA

p = 2
k = 3
N = 20

print(f'Generating a {k}-regular graph with {N} nodes')

G = nx.random_regular_graph(k, N)
G = nx.relabel_nodes(G, dict(zip(G.nodes, range(N))))

print('...done')

qaoa = QAOA(G, p)

init = [np.pi/8]*p + [0]*p

print(f'Starting optimization for p={p}')
angles, costs = qaoa.optimize(init=init, tol=1e-3)

print(f'Optimization done - cost reached: {costs[-1]}')
print(f'Angles: {angles}')

np.savetxt(f'angles_p{p}_N{N}.txt', angles, delimiter=', ')
nx.write_edgelist(G, f'edgelist_p{p}_N{N}.txt', data=False)