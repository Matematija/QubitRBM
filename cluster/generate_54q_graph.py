import numpy as np
import networkx as nx
import os, sys

libpath = os.path.abspath('..')
if libpath not in sys.path:
    sys.path.append(libpath)

from QubitRBM.qaoa import QAOA

nq = 54
k = 3

G = nx.random_regular_graph(k, nq)

qaoa = QAOA(G)
par, _ = qaoa.optimize(init=[np.pi/8, 0], tol=1e-5)
gamma, beta = par

np.savez('p1_opt_54_graph', graph=nx.to_numpy_array(G, nodelist=sorted(G.nodes)),
            gamma=gamma, beta=beta)