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

gamma_0, beta_0, gamma_1, beta_1 = 0.18634845102482034, -0.2425280149540639, 0.19102330493164887, -0.2825861551837291

logpsi = RBM(n_visible=nq)

G = nx.random_regular_graph(k, nq) if r==0 else None
G = comm.bcast(G, root=0)

if r==0:
    data = {'gamma_0': gamma_0, 'beta_0': beta_0,
            'gamma_1': gamma_1, 'beta_1': beta_1,
            'graph': nx.to_numpy_array(G, nodelist=sorted(G.nodes))}

    key_template = 'p{}#after_q{}#{}'

lr = 1e-1
tol = 1e-3

for i, j in G.edges(): 
    logpsi.RZZ(i, j, phi=2*gamma_0)

logpsi.mask[:] = True

for n in range(nq):
    
    if r == 0:
        print('Qubit {} starting at p=1...'.format(n+1))
        
    res = parallel_rx_optimization(comm, logpsi, n, beta_0, tol=tol, lr=lr, lookback=10, resample_phi=5,
                                    proc_psi_mcmc_params=(3000,2,1000,30), proc_phi_mcmc_params=(3000,2,1000,30),
                                    eps=1e-5, verbose=True)
    
    if r==0:
        params, Fs = res

        logpsi.set_flat_params(params)
        logpsi.fold_imag_params()
    
        data[key_template.format(1, n+1, 'C')] = logpsi.C.copy()
        data[key_template.format(1, n+1, 'a')] = logpsi.a.copy()
        data[key_template.format(1, n+1, 'b')] = logpsi.b.copy()
        data[key_template.format(1, n+1, 'W')] = logpsi.W.copy()
        data[key_template.format(1, n+1, 'Fs')] = Fs.copy()

        np.savez('output_data_54q_p2', **data)

    logpsi = comm.bcast(logpsi, root=0)

for i, j in G.edges(): 
    logpsi.RZZ(i, j, phi=2*gamma_1)

logpsi.mask[:] = True

for n in range(nq):
    
    if r == 0:
        print('Qubit {} starting at p=2...'.format(n+1))
        
    res = parallel_rx_optimization(comm, logpsi, n, beta_1, tol=tol, lr=lr, lookback=10, resample_phi=5,
                                    proc_psi_mcmc_params=(3000,2,1000,30), proc_phi_mcmc_params=(3000,2,1000,30),
                                    eps=1e-5, verbose=True)
    
    if r==0:
        params, Fs = res

        logpsi.set_flat_params(params)
        logpsi.fold_imag_params()
    
        data[key_template.format(2, n+1, 'C')] = logpsi.C.copy()
        data[key_template.format(2, n+1, 'a')] = logpsi.a.copy()
        data[key_template.format(2, n+1, 'b')] = logpsi.b.copy()
        data[key_template.format(2, n+1, 'W')] = logpsi.W.copy()
        data[key_template.format(2, n+1, 'Fs')] = Fs.copy()

        np.savez('output_data_54q_p2', **data)

    logpsi = comm.bcast(logpsi, root=0)

#### WRITING FILES ####

if r==0:
    np.savez('output_data_54q_p2', **data)