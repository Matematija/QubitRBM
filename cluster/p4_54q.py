import numpy as np
import sys, os
import networkx as nx

sys.path.append(os.path.abspath('..'))

from QubitRBM.optimize import rx_optimization, compress_rbm
from QubitRBM.rbm import RBM
from QubitRBM.qaoa import QAOA

p = 4
k = 3
nq = 54

compression_ps = [3, 4]
target_nh = 3*nq

data = {}

G20 = nx.random_regular_graph(k, 20)
while not nx.is_connected(G20):
    G20 = nx.random_regular_graph(k, 20)

qaoa20 = QAOA(G20, p=4)
data['graph_20'] = nx.to_numpy_array(G20, nodelist=sorted(G20.nodes))

print('\nStarting QAOA optimization for the 20 qubit graph...\n')
par, his = qaoa20.optimize(init=[np.pi/8]*p + [0]*p, tol=1e-3)
print('QAOA optimization done. Final cost: {}'.format(his[-1]))

gammas, betas = np.split(par, 2)
data['gammas'] = gammas
data['betas'] = betas

print('Gammas: ', gammas)
print('Betas: ', betas)

G = nx.random_regular_graph(k, nq)
while not nx.is_connected(G):
    G = nx.random_regular_graph(k, nq)

data['graph'] = nx.to_numpy_array(G, nodelist=sorted(G.nodes))

np.savez('p4_54q.npz', **data)

logpsi = RBM(n_visible=nq)

for i in range(p):

    print('Applying UC at p={}...'.format(i+1))

    for u, v in G.edges(): 
        logpsi.RZZ(u, v, phi=2*gammas[i])

    data['params_after_p{}_UC'.format(i+1)] = logpsi.params

    if i+1 in compression_ps:

        print('Starting compression at p={}...'.format(i+1))

        a = logpsi.a.copy()
        b = logpsi.b.copy()[:target_nh]
        W = logpsi.W.copy()[:,:target_nh]

        init = np.concatenate(a, b, W.reshape(-1))

        logpsi, Fs = compress_rbm(logpsi, target_hidden_num=target_nh, init=init, tol=1e-3, lookback=5, max_iters=100,
                                    psi_mcmc_params=(2000,4,500,54), phi_mcmc_params=(2000,4,500,54), sigma=0.0,
                                    resample_phi=5, lr=1e-1, eps=1e-4, verbose=True)

        logpsi.fold_imag_params()

        data['params_after_p{}_compression'.format(i+1)] = logpsi.params
        np.savez('p4_54q.npz', **data)

    print('Starting optimizations at p={}...'.format(i+1))

    for n in range(nq):
        
        print('Qubit {} starting at p={}...'.format(n+1, i+1))
            
        params, Fs = rx_optimization(rbm=logpsi, n=n, beta=betas[i], tol=1e-3, lr=1e-1, lookback=5, resample_phi=5, sigma=0.0,
                                    psi_mcmc_params=(2000,4,500,54), phi_mcmc_params=(2000,4,500,54), eps=1e-4, verbose=True)
        
        logpsi.params = params
        logpsi.fold_imag_params()

        data['params_after_q{}_p{}'.format(n+1, i+1)] = logpsi.params

    np.savez('p4_54q.npz', **data)