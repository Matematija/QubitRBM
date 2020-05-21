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

gamma_0, beta_0 = 1.3143397836447093, -0.4327704165659409 # p=1 optimal parameters

beta_1 = np.pi/8 # fixing \beta_1

gamma_1 = np.linspace(0, np.pi/4, size)[r]

logpsi = RBM(n_visible=20)
loaded = logpsi.load('rbm_params_20_qubit_optimal.npz')

G = nx.from_numpy_matrix(loaded['graph'])

for i, j in G.edges(): 
    logpsi.RZZ(i, j, phi=2*gamma_1)

# logpsi.mask[:] = True

data = {'gamma_0': gamma_0, 'beta_0': beta_0, 'gamma_1': gamma_1, 'beta_1': beta_1}
key_template = 'proc_{}#after_q{}#{}'

lr = 1e-1
tol = 1e-4

for n in range(nq):
    
    print('Qubit {} starting on process {}...'.format(n+1, r))
        
    params, Fs = rx_optimization(logpsi, n, beta_1, tol=tol, lr=lr, lookback=10, resample_phi=1, sigma=0.0,
                                   psi_mcmc_params=(2000,3,100,15), phi_mcmc_params=(2000,3,100,15),
                                   eps=1e-5, verbose=False)
    
    logpsi.set_flat_params(params)
    logpsi.fold_imag_params()
    
    data[key_template.format(r, n+1, 'C')] = logpsi.C
    data[key_template.format(r, n+1, 'a')] = logpsi.a
    data[key_template.format(r, n+1, 'b')] = logpsi.b
    data[key_template.format(r, n+1, 'W')] = logpsi.W
    data[key_template.format(r, n+1, 'Fs')] = Fs
        
    print('\nQubit {} done on process {}. Final fidelity estimate: {:05.4f}'.format(n+1, r, Fs[-1]))

#### WRITING FILES ####

save_folder = os.path.join(os.getcwd(), 'output_data_pi8')

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

save_path = os.path.join(save_folder, 'process_{}_data'.format(r))
np.savez(save_path, **data)