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

gamma_0, beta_0, _, beta_1 = 0.2393224253844294, -0.5350658391963277, 0.4409892830224139, -0.2820313757662484 # Optimal values for p=2 and k=3
gamma_1 = np.linspace(0, np.pi/2, size)[r]

logpsi = RBM(n_visible=20)
loaded = logpsi.load('rbm_params_20_qubit_optimal_p2_k3.npz') # k=3, preoptimized for p=1

G = nx.from_numpy_matrix(loaded['graph'])

for i, j in G.edges(): 
    logpsi.RZZ(i, j, phi=2*gamma_1)

# logpsi.add_hidden_units(num=4*logpsi.nv - logpsi.nh)
logpsi.mask[:] = True

data = {'gamma_0': gamma_0, 'beta_0': beta_0, 'gamma_1': gamma_1, 'beta_1': beta_1}
key_template = 'proc_{}#after_q{}#{}'

lr = 1e-1
tol = 1e-3

for n in range(nq):
    
    print('Qubit {} starting on process {}...'.format(n+1, r))
        
    params, Fs = rx_optimization(logpsi, n, beta_1, tol=tol, lr=lr, lookback=10, resample_phi=1, sigma=0.0,
                                   psi_mcmc_params=(2000,16,2000,30), phi_mcmc_params=(2000,16,2000,30),
                                   eps=1e-5, verbose=False)
    
    logpsi.set_flat_params(params)
    logpsi.fold_imag_params()
    
    data[key_template.format(r, n+1, 'C')] = logpsi.C.copy()
    data[key_template.format(r, n+1, 'a')] = logpsi.a.copy()
    data[key_template.format(r, n+1, 'b')] = logpsi.b.copy()
    data[key_template.format(r, n+1, 'W')] = logpsi.W.copy()
    data[key_template.format(r, n+1, 'Fs')] = Fs.copy()
        
    print('\nQubit {} done on process {}. Final fidelity estimate: {:05.4f}'.format(n+1, r, Fs[-1]))

#### WRITING FILES ####

save_folder = os.path.join(os.getcwd(), 'output_data_p2_opt')

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

save_path = os.path.join(save_folder, 'process_{}_data'.format(r))
np.savez(save_path, **data)