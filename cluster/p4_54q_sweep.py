import numpy as np
import sys, os
import networkx as nx
from collections import OrderedDict
from copy import deepcopy

sys.path.append(os.path.abspath('..'))

from QubitRBM.optimize import Optimizer
from QubitRBM.rbm import RBM

try:    
    N_JOBS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 10))
    print('Environment variable SLURM_ARRAY_TASK_COUNT = {}'.format(N_JOBS))
except:
    N_JOBS = 10

r = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
assert r is not None, 'Environment variable "SLURM_ARRAY_TASK_ID" not found!'

print('Slurm array task ID = {}'.format(r))

nq = 54
k = 3

print('Reading the file...')

f = np.load('p4_54q_after_p3')

params = f['params'].copy()
graph = nx.from_numpy_array(f['graph'])
betas = f['betas'].copy().item()
gammas = f['gammas'].copy().item()
beta_opt = betas[-1]
gamma_opt = gammas[-1]

gamma = (np.pi/4)*((r+1)/N_JOBS)

print('Optimal gamma:', gamma_opt)
print('Current gamma:', gamma)

n_hidden = (len(params) - nq)//(nq + 1)

logpsi = RBM(n_visible=nq, n_hidden=n_hidden)
logpsi.params = params

print('Hidden unit density before UC:', logpsi.alpha)

logpsi.UC(graph=graph, gamma=gamma, mask=False)
logpsi.mask[:] = True
logpsi.fold_imag_params()

mcmc_params = OrderedDict(zip(['n_steps', 'n_chains', 'warmup', 'step'], [2000, 4, 500, nq]))
optim = Optimizer(logpsi, **mcmc_params)

data = {'gamma': gamma, 'beta': beta_opt}
key_template = 'proc_{}#after_q{}#{}'

print('Beginning compression...')

aux = RBM(n_visible=nq)
aux.UC(G, gammas[:i+1].sum(), mask=False)
aux.mask[:] = True
init = aux.params.copy()

lri = 5e-1
lrf = 1e-1
tau_steps = 40
lr_tau = tau_steps/np.log(lri/lrf)
lr_min = lrf

logpsi_, Fs = optim.sr_compress(init=init, tol=1e-3, lookback=5, max_iters=1000, resample_phi=5, 
                                    r=lri, lr_tau=lr_tau, lr_min=lrf, eps=1e-4, verbose=True)
logpsi_.fold_imag_params()

logpsi = deepcopy(logpsi_)
logpsi.mask[:] = True
optim.machine = logpsi

data['params_after_p4_compression'] = logpsi.params

for n in range(nq):
    
    print('Qubit {} starting on process {}...'.format(n+1, r))
        
    params, Fs = optim.sr_rx(n, beta_opt, tol=1e-3, lookback=3, max_iters=1000, resample_phi=5, lr=1e-1, eps=1e-4, verbose=True)
    
    logpsi.params = params
    logpsi.fold_imag_params()
    
    data['params_after_q{}_p4'.format(n+1)]
    data[key_template.format(r, n+1, 'Fs')] = Fs.copy()

save_folder = os.path.join(os.getcwd(), 'output_data_54q_p1_sweep')

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

save_path = os.path.join(save_folder, 'process_{}_data'.format(r))
np.savez(save_path, **data)

# # Verifying the cost:
# final_samples = logpsi.get_samples(n_steps=10000, warmup=5000, step=54, n_chains=5)
# costs = ((-1)**final_samples[:, G.edges()]).prod(axis=-1).sum(axis=-1)
# print('Final cost estimate: {}'.format(costs.mean()))