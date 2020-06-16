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

nq = 54
k = 3

if r==0:
    f = np.load('54q_p1_optimal_graph_params_final.npz')
    G = nx.from_numpy_matrix(f['graph'])
    _, beta = f['params']
else:
    G, gamma_opt, beta = None, None, None

G = comm.bcast(G, root=0)
beta = comm.bcast(beta, root=0)

gamma = np.linspace(0, np.pi/2, size)[r]

logpsi = RBM(n_visible=nq)

for i, j in G.edges(): 
    logpsi.RZZ(i, j, phi=2*gamma)

logpsi.mask[:] = True

data = {'gamma': gamma, 'beta': beta}
key_template = 'proc_{}#after_q{}#{}'

lr = 1e-1
tol = 1e-3

for n in range(nq):
    
    print('Qubit {} starting on process {}...'.format(n+1, r))
        
    params, Fs = rx_optimization(logpsi, n, beta, tol=tol, lr=lr, lookback=5, resample_phi=5, sigma=0.0,
                                   psi_mcmc_params=(2000,16,1000,30), phi_mcmc_params=(2000,16,1000,30),
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

save_folder = os.path.join(os.getcwd(), 'output_data_54q_p1_sweep')

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

save_path = os.path.join(save_folder, 'process_{}_data'.format(r))
np.savez(save_path, **data)