import numpy as np
import sys, os
import networkx as nx
from collections import OrderedDict
from copy import deepcopy

sys.path.append(os.path.abspath('..'))

# from QubitRBM.optimize import rx_optimization, compress_rbm
from QubitRBM.rbm import RBM
from QubitRBM.optimize import Optimizer
from QubitRBM.qaoa import QAOA

p = 4
k = 3
nq = 54
mcmc_params = OrderedDict(zip(['n_steps', 'n_chains', 'warmup', 'step'], [2000, 4, 500, nq]))

data = OrderedDict(np.load('p4_54q.npz'))

G = nx.from_numpy_array(data['graph'])
gammas = data['gammas']
betas = data['betas']

logpsi = RBM(n_visible=nq, n_hidden=k*nq//2 +1)
logpsi.params = data['params_after_p2_compression'].copy()

optim = Optimizer(logpsi, **mcmc_params)

print('Beginning optimizations...')

for i in [1,2,3]:

    if i in [2, 3]:

        print('Applying UC at p={}...'.format(i+1))

        logpsi.UC(G, gammas[i], mask=False)
        logpsi.mask[:] = True

        data['params_after_p{}_UC'.format(i+1)] = logpsi.params

        print('Starting compression at p={}...'.format(i+1))

        aux = RBM(n_visible=nq)
        aux.UC(G, gammas[:i+1].sum(), mask=False)
        aux.mask[:] = True
        init = aux.params.copy()

        logpsi_, Fs = optim.sr_compress(init=init, tol=1e-3, lookback=5, max_iters=1000, resample_phi=5, lr=1e-1, eps=1e-4, verbose=True)
        logpsi_.fold_imag_params()

        logpsi = deepcopy(logpsi_)
        logpsi.mask[:] = True
        optim.machine = logpsi

        data['params_after_p{}_compression'.format(i+1)] = logpsi.params
        data['fidelities_on_p{}_compression'.format(i+1)] = Fs
        np.savez('p4_54q.npz', **data)

    print('Starting optimizations at p={}...'.format(i+1))

    for n in range(nq):
        
        print('Qubit {} starting at p={}...'.format(n+1, i+1))
            
        params, Fs = optim.sr_rx(n, betas[i], tol=1e-3, lookback=3, max_iters=1000, resample_phi=5, lr=1e-1, eps=1e-4, verbose=True)
        
        logpsi.params = params
        logpsi.fold_imag_params()

        data['params_after_q{}_p{}'.format(n+1, i+1)] = logpsi.params

    np.savez('p4_54q.npz', **data)