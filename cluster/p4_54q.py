import numpy as np
import sys, os
import networkx as nx
from collections import OrderedDict

sys.path.append(os.path.abspath('..'))

# from QubitRBM.optimize import rx_optimization, compress_rbm
from QubitRBM.rbm import RBM
from QubitRBM.optimize import Optimizer
from QubitRBM.qaoa import QAOA

p = 4
k = 3
nq = 54
mcmc_params = OrderedDict(zip(['n_steps', 'n_chains', 'warmup', 'step'], [2000, 4, 500, nq]))

data = OrderedDict()

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

print('Beginning optimizations...')

for i in range(p):

    print('Applying UC at p={}...'.format(i+1))

    logpsi.UC(G, gammas[i], mask=False)
    logpsi.mask[:] = True

    data['params_after_p{}_UC'.format(i+1)] = logpsi.params

    optim = Optimizer(logpsi, **mcmc_params)

    if i > 0:

        print('Starting compression at p={}...'.format(i+1))

        aux = RBM(n_visible=nq)
        aux.UC(G, gammas[:i+1].sum(), mask=False)
        init = aux.params

        logpsi, Fs = optim.sr_compress(init=init, tol=1e-3, lookback=5, max_iters=1000, resample_phi=5, lr=1e-1, eps=1e-4, verbose=True)

        logpsi.fold_imag_params()

        data['params_after_p{}_compression'.format(i+1)] = logpsi.params
        np.savez('p4_54q.npz', **data)

    print('Starting optimizations at p={}...'.format(i+1))

    for n in range(nq):
        
        print('Qubit {} starting at p={}...'.format(n+1, i+1))
            
        params, Fs = optim.sr_rx(n, betas[i], tol=1e-3, lookback=3, max_iters=1000, resample_phi=5, lr=1e-1, eps=1e-4, verbose=True)
        
        logpsi.params = params
        logpsi.fold_imag_params()

        data['params_after_q{}_p{}'.format(n+1, i+1)] = logpsi.params

    np.savez('p4_54q.npz', **data)