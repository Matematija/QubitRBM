import numpy as np
import sys, os
import networkx as nx

sys.path.append(os.path.abspath('..'))

from QubitRBM.optimize import *
from QubitRBM.rbm import *

from mpi4py import MPI

comm = MPI.COMM_WORLD
r = comm.Get_rank()
size = comm.Get_size()

nq = 20
k = 3

gamma_opt, beta = 0.28851361104396056, -0.36865628077839413 # Optimal values for p=1 and k=3
gamma = np.linspace(0, np.pi/2, size)[r]

G = nx.random_regular_graph(k, nq)

logpsi = RBM(n_visible=nq)

for i, j in G.edges(): 
    logpsi.RZZ(i, j, phi=2*gamma)

# logpsi.add_hidden_units(num=4*logpsi.nv - logpsi.nh)
logpsi.mask[:] = True

data = {'gamma_0': gamma, 'beta_0': beta, 'graph': nx.to_numpy_array(G, nodelist=sorted(G.nodes))}
key_template = 'proc_{}#after_q{}#{}'

lr = 1e-1
tol = 1e-3

for n in range(nq):
    
    print('Qubit {} starting on process {}...'.format(n+1, r))
        
    params, Fs = rx_optimization(logpsi, n, beta, tol=tol, lr=lr, lookback=10, resample_phi=1, sigma=0.0,
                                   psi_mcmc_params=(2000,16,200,20), phi_mcmc_params=(2000,16,200,20),
                                   eps=1e-5, verbose=False)
    
    logpsi.set_flat_params(params)
    logpsi.fold_imag_params()
    
    data[key_template.format(r, n+1, 'C')] = logpsi.C.copy()
    data[key_template.format(r, n+1, 'a')] = logpsi.a.copy()
    data[key_template.format(r, n+1, 'b')] = logpsi.b.copy()
    data[key_template.format(r, n+1, 'W')] = logpsi.W.copy()
    data[key_template.format(r, n+1, 'Fs')] = np.array(Fs).copy()
        
    print('\nQubit {} done on process {}. Final fidelity estimate: {:05.4f}'.format(n+1, r, Fs[-1]))

#### WRITING FILES ####

save_folder = os.path.join(os.getcwd(), 'output_data_p1_opt')

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

save_path = os.path.join(save_folder, 'process_{}_data'.format(r))
np.savez(save_path, **data)