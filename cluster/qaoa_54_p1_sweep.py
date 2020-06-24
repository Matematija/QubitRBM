import numpy as np
import sys, os
import networkx as nx
from math import floor

sys.path.append(os.path.abspath('..'))

from QubitRBM.optimize import *
from QubitRBM.rbm import *
from QubitRBM.qaoa import QAOA

from mpi4py import MPI

comm = MPI.COMM_WORLD
r = comm.Get_rank()
size = comm.Get_size()

nq = 54
k = 3

if r==0:
    G = nx.random_regular_graph(k, nq)
    qaoa = QAOA(G)

    par, _ = qaoa.optimize(init=[np.pi/8, 0], tol=1e-5)
    gamma_opt, beta_opt = par

    print('Optimal parameters for the generated graph:')
    print('gamma = {}'.format(gamma_opt))
    print('beta = {}'.format(beta_opt))

else:
    G, gamma_opt, beta_opt = None, None, None

G = comm.bcast(G, root=0)
beta_opt = comm.bcast(beta_opt, root=0)

gamma = np.linspace(0, np.pi/2, size)[r]

logpsi = RBM(n_visible=nq)

for i, j in G.edges(): 
    logpsi.RZZ(i, j, phi=2*gamma)

alpha = 2.2
logpsi.add_hidden_units(num=floor(alpha*logpsi.nv - logpsi.nh))

logpsi.mask[:] = True

data = {'gamma': gamma, 'beta': beta_opt}
key_template = 'proc_{}#after_q{}#{}'

save_folder = os.path.join(os.getcwd(), 'output_data_54q_p1_sweep')

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

if r==0:
    graph_save_path = os.path.join(save_folder, 'graph')
    np.save(graph_save_path, nx.to_numpy_array(G, nodelist=sorted(G.nodes)))

lr = 1e-1
tol = 1e-3

for n in range(nq):
    
    print('Qubit {} starting on process {}...'.format(n+1, r))
        
    params, Fs = rx_optimization(logpsi, n, beta_opt, tol=tol, lr=lr, lookback=3, resample_phi=5, sigma=0.0,
                                   psi_mcmc_params=(1500,4,500,54), phi_mcmc_params=(1500,4,500,54),
                                   eps=1e-5, verbose=False)
    
    logpsi.params = params
    logpsi.fold_imag_params()
    
    data[key_template.format(r, n+1, 'C')] = logpsi.C.copy()
    data[key_template.format(r, n+1, 'a')] = logpsi.a.copy()
    data[key_template.format(r, n+1, 'b')] = logpsi.b.copy()
    data[key_template.format(r, n+1, 'W')] = logpsi.W.copy()
    data[key_template.format(r, n+1, 'Fs')] = Fs.copy()
        
    print('\nQubit {} done on process {}. Final fidelity estimate: {:05.4f}'.format(n+1, r, Fs[-1]))

#### WRITING FILES ####

save_path = os.path.join(save_folder, 'process_{}_data'.format(r))
np.savez(save_path, **data)