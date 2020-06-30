import numpy as np
import sys, os
import networkx as nx
from math import floor

sys.path.append(os.path.abspath('..'))

from QubitRBM.optimize import rx_optimization
from QubitRBM.rbm import RBM

r = sys.getenv('SLURM_ARRAY_TASK_ID')

nq = 54
k = 3

f = np.load('p1_opt_54_graph.npz')
G = nx.from_numpy_array(f['graph'])
beta_opt = f['beta'].item()
gamma = np.linspace(0, np.pi/2, size)[r]

logpsi = RBM(n_visible=nq)

for i, j in G.edges(): 
    logpsi.RZZ(i, j, phi=2*gamma)

logpsi.fold_imag_params()

alpha = 2
logpsi.add_hidden_units(num=floor(alpha*logpsi.nv) - logpsi.nh)
logpsi.mask[:] = True

data = {'gamma': gamma, 'beta': beta_opt}
key_template = 'proc_{}#after_q{}#{}'

for n in range(nq):
    
    print('Qubit {} starting on process {}...'.format(n+1, r))
        
    params, Fs = rx_optimization(rbm=logpsi, n=n, beta=beta_opt, tol=1e-3, lr=1e-1, lookback=3, resample_phi=5, sigma=0.0,
                                   psi_mcmc_params=(1500,4,500,54), phi_mcmc_params=(1500,4,500,54),
                                   eps=1e-5, verbose=False)
    
    logpsi.params = params
    logpsi.fold_imag_params()
    
    data[key_template.format(r, n+1, 'C')] = logpsi.C.copy()
    data[key_template.format(r, n+1, 'a')] = logpsi.a.copy()
    data[key_template.format(r, n+1, 'b')] = logpsi.b.copy()
    data[key_template.format(r, n+1, 'W')] = logpsi.W.copy()
    data[key_template.format(r, n+1, 'Fs')] = Fs.copy()
        
    print('\nQubit {} done on process {}. Final fidelity estimate: {:05.4f}'.format(n+1, r, Fs[-1]))

save_folder = os.path.join(os.getcwd(), 'output_data_54q_p1_sweep')

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

save_path = os.path.join(save_folder, 'process_{}_data'.format(r))
np.savez(save_path, **data)