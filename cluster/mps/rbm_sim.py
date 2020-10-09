import numpy as np
import sys, os
import networkx as nx
from collections import OrderedDict
from copy import deepcopy

path = os.path.abspath('../../')
sys.path.append(path)

from qubitrbm.rbm import RBM
from qubitrbm.optimize import Optimizer
from qubitrbm.qaoa import QAOA

assert len(sys.argv) == 3, 'Not enough params set'

edgelist_path = sys.argv[1]
angles_path = sys.argv[2]

G = nx.read_edgelist(edgelist_path, data=False, nodetype=int)
gammas, betas = np.split(np.loadtxt(angles_path, delimiter=', '), 2)

N = len(G.nodes)
p = len(gammas)
k = G.degree[0]

print(f'Read in a graph with N={N}, k={k}')
print(f'Depth p={p}')
print(f'gammas = {gammas}')
print(f'betas = {betas}', flush=True)

mcmc_params = OrderedDict(zip(['n_steps', 'n_chains', 'warmup', 'step'], [2000, 4, 500, N]))

print('MCMC params set:')
print('\n'.join([f'{k}={v}' for k,v in mcmc_params.items()]), flush=True)

logpsi = RBM(n_visible=N, n_hidden=k*N//2 +1)
optim = Optimizer(logpsi, **mcmc_params)

data = OrderedDict()

print('Beginning optimizations...', flush=True)

for i in range(p):

    print(f'Applying UC at p={i+1}', flush=True)

    logpsi.UC(G, gammas[i], mask=False)
    logpsi.mask[:] = True

    data['params_after_p{i+1}_UC'] = logpsi.params

    if i > 0:
        print(f'Starting compression at p={i+1}...', flush=True)

        aux = RBM(n_visible=N)
        aux.UC(G, gammas[:i+1].sum(), mask=False)
        aux.mask[:] = True
        init = aux.params.copy()

        logpsi_, Fs = optim.sr_compress(init=init, tol=1e-3, lookback=5,
                                        max_iters=1000, resample_phi=5,
                                        lr=1e-1, eps=1e-4, verbose=True)
        logpsi_.fold_imag_params()

        logpsi = deepcopy(logpsi_)
        logpsi.mask[:] = True
        optim.machine = logpsi

        data['params_after_p{i+1}_compression'] = logpsi.params
    
    for n in range(N):

        print(f'Qubit {n+1} starting at p={i+1}...', flush=True)
            
        params, Fs = optim.sr_rx(n, betas[i], tol=1e-3, lookback=3, max_iters=1000,
                                    resample_phi=5, lr=1e-1, eps=1e-4, verbose=True)
        
        logpsi.params = params
        logpsi.fold_imag_params()

        data[f'params_after_q{n+1}_p{i+1}'] = logpsi.params

np.savez('rbm_res_N{}_p{}.npz', **data)

samples = logpsi.get_samples(n_steps=2000, n_chains=10, step=N, warmup=2000)
costs = QAOA(G, p).cost(samples)

print(f'Mean cost: {costs.mean()}')