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

if r==0:
    print('Running {} processes'.format(size))

nq = 54
k = 3

gamma, beta = 0.28851361104396056, -0.36865628077839413 # Optimal values for p=1 and k=3

logpsi = RBM(n_visible=nq)

G = nx.random_regular_graph(k, nq) if r==0 else None
G = comm.bcast(G, root=0)

for i, j in G.edges(): 
    logpsi.RZZ(i, j, phi=2*gamma)

# logpsi.add_hidden_units(num=4*logpsi.nv - logpsi.nh)
logpsi.mask[:] = True

if r==0:
    data = {'gamma_0': gamma, 'beta_0': beta, 'graph': nx.to_numpy_array(G, nodelist=sorted(G.nodes))}
    key_template = 'after_q{}#{}'

lr = 1e-1
tol = 1e-3

for n in range(nq):
    
    if r == 0:
        print('Qubit {} starting...'.format(n+1))
        
    res = parallel_rx_optimization(comm, logpsi, n, beta, tol=tol, lr=lr, lookback=10, resample_phi=3,
                                    proc_psi_mcmc_params=(5000,2,1000,30), proc_phi_mcmc_params=(5000,2,1000,30),
                                    eps=1e-5, verbose=True)
    
    if r==0:
        params, Fs = res

        logpsi.set_flat_params(params)
        logpsi.fold_imag_params()
    
        data[key_template.format(n+1, 'C')] = logpsi.C.copy()
        data[key_template.format(n+1, 'a')] = logpsi.a.copy()
        data[key_template.format(n+1, 'b')] = logpsi.b.copy()
        data[key_template.format(n+1, 'W')] = logpsi.W.copy()
        data[key_template.format(n+1, 'Fs')] = Fs.copy()

    logpsi = comm.bcast(logpsi, root=0)

#### WRITING FILES ####

if r==0:
    save_folder = os.path.join(os.getcwd(), 'output_data_54_p1')

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    save_path = os.path.join(save_folder, 'process_{}_data'.format(r))
    np.savez(save_path, **data)