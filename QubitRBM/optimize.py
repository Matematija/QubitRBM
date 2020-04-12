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

def hadamard_optimization(rbm, n, tol=1e-6, lookback=50, psi_mcmc_params=(500, 100, 1), phi_mcmc_params=(2000, 500, 1),
                           sigma=1e-3, resample_phi=None, lr=0.05, lr_tau=None, lr_min=0.0, eps=1e-6, fidelity='mcmc', verbose=False):

    """
    Implements the stochastic reconfiguration algorithm to optimize the application of one H gate on qubit n in machine 'rbm'.
    """

    psi_mcmc_steps, psi_warmup, psi_gap = psi_mcmc_params
    phi_mcmc_steps, phi_warmup, phi_gap = phi_mcmc_params
    nv, nh = rbm.nv, rbm.nh

    if fidelity == 'exact':
        H = eg.H(n=n, n_qubits=nv)
        psi = rbm.get_state_vector(normalized=True)
        target_state = H.dot(psi)
    elif fidelity == 'mcmc':
        pass
    else:
        raise KeyError('Invalid fidelity calculation mode. Expected "exact" or "mcmc", got {}'.format(fidelity))
        
    phi_samples = rbm.get_samples(n_steps=phi_gap*phi_mcmc_steps+phi_warmup+1, state='h', n=n)[phi_warmup::phi_gap]

    a = rbm.a.copy() + sigma*(np.random.randn(nv) + 1j*np.random.randn(nv))
    b = rbm.b.copy() + sigma*(np.random.randn(nh) + 1j*np.random.randn(nh))
    W = rbm.W.copy() + sigma*(np.random.randn(nv, nh) + 1j*np.random.randn(nv, nh))

    logpsi = RBM(nv, nh)
    logpsi.set_params(a=a, b=b, W=W)
    params = utils.pack_params(a, b, W)
    
    history = []
    F = 0
    F_mean_new = 0.0
    F_mean_old = 0.0
    lr_ = lr
    clock = time()
    t = 0
    
    while (np.abs(F_mean_new - F_mean_old) > tol or t < 2*lookback + 1) and F_mean_new < 1-tol:
    # while F_mean_new < 1 - tol or t < 2*lookback + 1:

        t += 1

        psi_samples = logpsi.get_samples(n_steps=psi_gap*psi_mcmc_steps+psi_warmup+1)[psi_warmup::psi_gap]

        phipsi = rbm.eval_H(n, psi_samples)
        psipsi = logpsi(psi_samples)
    
        if fidelity == 'exact': 
            psi_vec = logpsi.get_state_vector(normalized=True)
            F = utils.exact_fidelity(psi_vec, target_state)
        elif fidelity == 'mcmc':
            psiphi = logpsi(phi_samples)
            phiphi = rbm.eval_H(n, phi_samples)
            F = utils.mcmc_fidelity(psipsi, psiphi, phipsi, phiphi)

        history.append(F)

        if t > 2*lookback:
            F_mean_old = F_mean_new
            F_mean_new = sum(history[-lookback:])/lookback

        gas, gbs, gWs = logpsi.grad_log(psi_samples)
        O = np.concatenate([gas, gbs, gWs.reshape(-1, nv*nh)], axis=1)
        S = S_matrix(O)

        ratio_psi = np.exp(phipsi - psipsi)
        ratio_psi_mean = ratio_psi.mean()

        grad_logF = O.mean(axis=0).conj() - (ratio_psi.reshape(-1,1)*O.conj()).mean(axis=0)/ratio_psi_mean

        grad = F*grad_logF
        delta_theta = solve(S + eps*np.eye(S.shape[0]), grad, assume_a='her')
        
        if lr_tau is not None:
            lr_ = max(lr_min, lr*np.exp(-t/lr_tau))

        params -= lr_*delta_theta

        logpsi.a, logpsi.b, logpsi.W = utils.unpack_params(params, nv, nh)
        
        if resample_phi is not None:
            if t%resample_phi == 0:
                phi_samples = rbm.get_samples(n_steps=phi_gap*phi_mcmc_steps+phi_warmup+1, state='h', n=n)[phi_warmup::phi_gap]

        if time() - clock > 20 and verbose:
            print('Iteration {} | Fidelity = {} | lr = {}'.format(t, F, lr_))
            clock = time()

    return logpsi.a, logpsi.b, logpsi.W, history