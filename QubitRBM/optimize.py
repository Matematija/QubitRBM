import numpy as np
from scipy.linalg import solve
from scipy.special import logsumexp

from time import time
from copy import deepcopy
import os, sys

libpath = os.path.abspath('..')
if libpath not in sys.path:
    sys.path.append(libpath)

from QubitRBM.rbm import RBM
import QubitRBM.exact_gates as eg
import QubitRBM.utils as utils

def S_matrix(O):
    O_centered = O - O.mean(axis=0, keepdims=True)
    return np.matmul(O_centered.T.conj(), O_centered)/O.shape[0]

def rx_optimization(rbm, n, beta, tol=1e-6, lookback=50, psi_mcmc_params=(500,5,50,1), phi_mcmc_params=(500,5,50,1),
                    sigma=1e-5, resample_phi=None, lr=5e-2, lr_tau=None, lr_min=0.0, eps=1e-6, verbose=False):

    psi_mcmc_args = dict(zip(['n_steps', 'n_chains', 'warmup', 'step'], psi_mcmc_params))
    phi_mcmc_args = dict(zip(['n_steps', 'n_chains', 'warmup', 'step'], phi_mcmc_params))
    nv, nh = rbm.nv, rbm.nh

    logpsi = deepcopy(rbm)
    
    if np.abs(np.sin(beta)) > np.abs(np.cos(beta)):
        logpsi.X(n)
    
    params = logpsi.get_flat_params()
    phi_samples = rbm.get_samples(**phi_mcmc_args, state='rx', n=n, beta=beta)
    
    phiphi = rbm.eval_RX(phi_samples, n=n, beta=beta)
    
    history = []
    F = 0
    F_mean_new = 0.0
    F_mean_old = 0.0
    lr_ = lr
    clock = time()
    t = 0

    while (np.abs(F_mean_new - F_mean_old) > tol or t < 2*lookback + 1) and F_mean_new < 0.99:
        
        t += 1

        psi_samples = logpsi.get_samples(**psi_mcmc_args)
        
        psipsi = logpsi(psi_samples)
        phipsi = rbm.eval_RX(psi_samples, n=n, beta=beta)
        psiphi = logpsi(phi_samples)
        
        F = utils.mcmc_fidelity(psipsi, psiphi, phipsi, phiphi)

        history.append(F)

        if t > 2*lookback:
            F_mean_old = sum(history[-2*lookback:-lookback])/lookback
            F_mean_new = sum(history[-lookback:])/lookback

        O = logpsi.grad_log(psi_samples)
        S = S_matrix(O)

        ratio_psi = np.exp(phipsi - psipsi)
        ratio_psi_mean = ratio_psi.mean()

        grad_logF = O.mean(axis=0).conj() - (ratio_psi.reshape(-1,1)*O.conj()).mean(axis=0)/ratio_psi_mean
        grad = F*grad_logF

        S[np.diag_indices_from(S)] += eps 
        delta_theta = solve(S, grad, assume_a='her')
        
        if lr_tau is not None:
            lr_ = max(lr_min, lr*np.exp(-t/lr_tau))

        params -= lr_*delta_theta
        logpsi.set_flat_params(params)
        
        if resample_phi is not None:
            if t%resample_phi == 0:
                phi_samples = rbm.get_samples(**phi_mcmc_args, state='rx', beta=beta, n=n)
                phiphi = rbm.eval_RX(phi_samples, n=n, beta=beta)

        if time() - clock > 5 and verbose:
            diff_mean_F = np.abs(F_mean_new - F_mean_old)
            print('Iteration {:4d} | Fidelity = {:05.4f} | lr = {:04.3f} | diff_mean_F = {:08.7f}'.format(t, F, lr_, diff_mean_F))
            clock = time()

    return params, history