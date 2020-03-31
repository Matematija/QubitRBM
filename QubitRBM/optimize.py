import numpy as np
from scipy.linalg import solve
from scipy.special import logsumexp

from time import time

import os, sys
libpath = os.path.abspath('..')
if libpath not in sys.path:
    sys.path.append(libpath)

try:
    from QubitRBM.rbm import RBM
    import QubitRBM.exact_gates as eg
    import QubitRBM.utils as utils
except Exception as error:
    print('QubitRBM folder not in PATH!')
    raise(error)


def S_matrix(grad):
    term1 = np.matmul(grad.conj().T, grad)/grad.shape[0]
    term2 = np.tensordot(grad.conj().mean(axis=0), grad.mean(axis=0), axes=0)
    return term1 - term2

def hadamard_optimization(rbm, n, tol=1e-5, lookback=10, mcmc_params=(500, 100, 1), sigmas=(1e-3, 1e-3, 1e-3),
    lr_init=1e-2, lr_tau=None, eps=1e-6, fidelity='mcmc', verbose=False):

    """
    Implements the stochastic reconfiguration algorithm to optimize the application of one H gate on qubit n in machine 'rbm'.
    """
    
    mcmc_steps, warmup, gap = mcmc_params
    sigma_a, sigma_b, sigma_W = sigmas
    nv, nh = rbm.nv, rbm.nh
    
    if fidelity == 'exact':
        H = eg.H(n=n, n_qubits=nv)
        psi = rbm.get_state_vector(normalized=True, log=False)
        target_state = H.dot(psi)

    elif fidelity == 'mcmc':
        pass
    else:
        raise KeyError('Invalid fidelity calculation mode. Expected "exact" or "mcmc", got {}'.format(fidelity))
    
    init = np.random.rand(rbm.nv) > 0.5
    phi_samples = rbm.get_samples(init=init, n_steps=mcmc_steps, state='h', n=n)
    
    if lr_tau is None:
        lr_tau = 1000/np.log(lr_init/1e-4)
    
    a = rbm.a.copy() + sigma_a*(np.random.randn(nv) + 1j*np.random.randn(nv))
    b = rbm.b.copy() + sigma_b*(np.random.randn(nh) + 1j*np.random.randn(nh))
    W = rbm.W.copy() + sigma_W*(np.random.randn(nv, nh) + 1j*np.random.randn(nv, nh))
    
    params = utils.pack_params(a, b, W)
    logpsi = RBM(nv, nh)
    logpsi.set_params(a, b, W)

    history = []
    F_mean_new = 0.0
    F_mean_old = 0.0
    clock = time()
    t = 0
    
    while np.abs(F_mean_new - F_mean_old) > tol or t < 2*lookback:
        
        t += 1

        init = np.random.rand(nv) > 0.5
        psi_samples = logpsi.get_samples(init, gap*mcmc_steps+warmup+1)[warmup::gap]

        phipsi = rbm.eval_hadamard(n, psi_samples)
        psipsi = logpsi(psi_samples)

        if fidelity == 'exact': 
            psi_vec = logpsi.get_state_vector(normalized=True, log=False)
            F = utils.exact_fidelity(psi_vec, target_state)
        elif fidelity == 'mcmc':
            psiphi = logpsi(phi_samples)
            phiphi = rbm.eval_hadamard(n, phi_samples)
            F = utils.mcmc_fidelity(psipsi, psiphi, phipsi, phiphi)
        history.append(F)
        
        if t > 2*lookback:
            F_mean_new = sum(history[-lookback:])/lookback
            F_mean_old = sum(history[-2*lookback:-lookback])/lookback

        gas, gbs, gWs = logpsi.grad_log(psi_samples)
        O = np.concatenate([gas, gbs, gWs.reshape(-1, nv*nh)], axis=1)
        S = S_matrix(O)
        
        ratio_psi = np.exp(phipsi - psipsi)
        ratio_psi_mean = ratio_psi.mean()

        grad_logF = O.mean(axis=0).conj() - (ratio_psi.reshape(-1,1)*O.conj()).mean(axis=0)/ratio_psi_mean
    
        grad = F*grad_logF
        delta_theta = solve(S + eps*np.eye(S.shape[0]), grad, assume_a='her')
        
        lr = lr_init*np.exp(-t/lr_tau)
        params -= lr*delta_theta
        
        logpsi.a, logpsi.b, logpsi.W = utils.unpack_params(params)

        if time() - clock > 10 and verbose:
            print('Iteration {} | Fidelity = {}'.format(t, F))
            clock = time()
            
    return logpsi.a, logpsi.b, logpsi.W, history