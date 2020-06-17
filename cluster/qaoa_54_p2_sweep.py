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
    logpsi = RBM(n_visible=nq)
    f = logpsi.load('54q_p2_opt_layer1_rbm_params.npz')
    G = nx.from_numpy_matrix(f['graph'])
else:
    G = None
    logpsi = None

G = comm.bcast(G, root=0)
logpsi = comm.bcast(logpsi, root=0)

gamma_1, beta_1 = 0.18634003777801475, -0.2425669908912676
_, beta_2 = 0.1910950814611548, -0.2826951383803437

gamma_2 = np.linspace(0, np.pi/2, size)[r]

for i, j in G.edges(): 
    logpsi.RZZ(i, j, phi=2*gamma_2)

logpsi.mask[:] = True

data = {'gamma_1': gamma_1, 'beta_1': beta_1, 'gamma_2': gamma_2, 'beta_2': beta_2,
        'graph': nx.to_numpy_array(G, nodelist=sorted(G.nodes))}

key_template = 'proc_{}#after_q{}#{}'

lr = 1e-1
tol = 1e-3

for n in range(nq):
    
    print('Qubit {} starting on process {}...'.format(n+1, r))
        
    params, Fs = rx_optimization(logpsi, n, beta_2, tol=tol, lr=lr, lookback=2, resample_phi=5, sigma=0.0,
                                   psi_mcmc_params=(2000,10,1000,30), phi_mcmc_params=(2000,10,1000,30),
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

save_folder = os.path.join(os.getcwd(), 'output_data_54q_p2_sweep')

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

save_path = os.path.join(save_folder, 'process_{}_data'.format(r))
np.savez(save_path, **data)