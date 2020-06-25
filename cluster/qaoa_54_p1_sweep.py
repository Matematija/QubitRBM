import numpy as np
import sys, os
import networkx as nx
from math import floor

sys.path.append(os.path.abspath('..'))

from QubitRBM.optimize import *
from QubitRBM.rbm import *

from mpi4py import MPI

comm = MPI.COMM_WORLD
r = comm.Get_rank()
size = comm.Get_size()

nq = 54
k = 3

def exact_qaoa_cost(graph, gamma, beta, deltas):

    C = 0.0
    s2g, c2g, c4g = np.sin(2*gamma), np.cos(2*gamma), np.cos(4*gamma)
    s4b, s2b = np.sin(4*beta), np.sin(2*beta)
    
    for (u, v), D in zip(graph.edges(), deltas):

        du = graph.degree(u) - 1
        dv = graph.degree(v) - 1
        
        term1 = s4b*s2g*(c2g**du + c2g**dv)
        term2 = (s2b**2)*(c2g**(du+dv-2*D))*(1 - c4g**D)

        C += (term1 + term2)/2
    
    return C

def exact_qaoa_grad_cost(graph, gamma, beta, deltas):

    d_beta, d_gamma = 0.0, 0.0

    c2g, c4g, s2g, s4g = np.cos(2*gamma), np.cos(4*gamma), np.sin(2*gamma), np.sin(4*gamma)
    s2b, s4b, c4b = np.sin(2*beta), np.sin(4*beta), np.cos(4*beta)

    for (k, l), D in zip(graph.edges(), deltas):
        
        dk = graph.degree(k) - 1
        dl = graph.degree(l) - 1

        gamma_term_1 = s4b * c2g * (c2g**dk + c2g**dl)
        gamma_term_2 = -(dk+dl-2*D) * s2b**2 * s2g * c2g**(dk+dl-2*D-1) * (1-c4g**D)
        gamma_term_3 = 2*D * (s2b**2) * s4g * c2g**(dk+dl-2*D) * c4g**(D-1)
        gamma_term_4 = -s4b * s2g**2 * (dk* c2g**(dk-1) + dl* c2g**(dl-1))

        d_gamma += gamma_term_1 + gamma_term_2 + gamma_term_3 + gamma_term_4

        beta_term_1 = 2 * c4b * s2g * (c2g**dk + c2g**dl)
        beta_term_2 = s4b * c2g**(dk+dl-2*D) * (1-c4g**D)

        d_beta +=  beta_term_1 + beta_term_2

    return np.array([d_gamma, d_beta])

def optimize_qaoa(graph, deltas, lr=1e-3, tol=1e-3, init=None, betas=(0.9, 0.999), eps=1e-6, dx=1e-5, verbose=True):

        beta1, beta2 = betas
        
        if init is None:
            gamma = np.random.uniform(0, np.pi/2)
            beta = np.random.uniform(-np.pi/4, np.pi/4)
            params = np.array([gamma, beta])
        else:
            params = np.array(init).copy()

        g, b = params

        m = np.zeros_like(params)
        v = np.zeros_like(params)
        t = 0
        history = []

        while True:

            t += 1

            grad = exact_qaoa_grad_cost(graph, g, b, deltas)

            m = beta1*m + (1-beta1)*grad
            v = beta2*v + (1-beta2)*grad**2

            m_hat = m/(1 + beta1**t)
            v_hat = v/(1 + beta2**t)

            d_params = lr * m_hat/(np.sqrt(v_hat) + eps)

            if np.any(d_params > tol):
                params -= d_params
            else:
                break

            g, b = params
            f = exact_qaoa_cost(graph, g, b, deltas)

            history.append(f)

        return params, np.array(history)

if r==0:
    G = nx.random_regular_graph(k, nq)

    triangles = [c for c in nx.enumerate_all_cliques(G) if len(c)==3]
    deltas = [len([t for t in triangles if u in t and v in t]) for u, v in G.edges()]

    par, _ = optimize_qaoa(G, deltas, init=[np.pi/8, 0], tol=1e-5)
    gamma_opt, beta_opt = par

    print('Optimal parameters for the generated graph:')
    print('gamma = {}'.format(gamma_opt))
    print('beta = {}'.format(beta_opt))

else:
    G, gamma_opt, beta_opt = None, None, None

G = comm.bcast(G, root=0)
beta_opt = comm.bcast(beta_opt, root=0)

gamma = np.linspace(0, np.pi/2, size)[r]

logpsi = RBM(n_visible=nq)

for i, j in G.edges(): 
    logpsi.RZZ(i, j, phi=2*gamma)

alpha = 2.2
logpsi.add_hidden_units(num=floor(alpha*logpsi.nv - logpsi.nh))

logpsi.mask[:] = True

data = {'gamma': gamma, 'beta': beta_opt}
key_template = 'proc_{}#after_q{}#{}'

save_folder = os.path.join(os.getcwd(), 'output_data_54q_p1_sweep')

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

if r==0:
    graph_save_path = os.path.join(save_folder, 'graph')
    np.save(graph_save_path, nx.to_numpy_array(G, nodelist=sorted(G.nodes)))

lr = 1e-1
tol = 1e-3

for n in range(nq):
    
    print('Qubit {} starting on process {}...'.format(n+1, r))
        
    params, Fs = rx_optimization(logpsi, n, beta_opt, tol=tol, lr=lr, lookback=3, resample_phi=5, sigma=0.0,
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

#### WRITING FILES ####

save_path = os.path.join(save_folder, 'process_{}_data'.format(r))
np.savez(save_path, **data)