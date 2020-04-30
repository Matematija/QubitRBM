import numpy as np
from scipy.linalg import solve
from scipy.special import logsumexp

from time import time
import os, sys

libpath = os.path.abspath('..')
if libpath not in sys.path:
    sys.path.append(libpath)

from QubitRBM.rbm import RBM
import QubitRBM.exact_gates as eg
import QubitRBM.utils as utils

def S_matrix(grad):
    term1 = np.matmul(grad.conj().T, grad)/grad.shape[0]

    grad_mean = grad.mean(axis=0)
    term2 = np.tensordot(grad_mean.conj(), grad_mean, axes=0)

    return term1 - term2

def hadamard_optimization(rbm, n, init=None, comm=None, tol=1e-6, lookback=50, psi_mcmc_params=(500, 100, 1), phi_mcmc_params=(2000, 500, 1),
                           sigma=1e-5, resample_phi=None, lr=0.05, lr_tau=None, lr_min=0.0, eps=1e-6, fidelity='mcmc', verbose=False):

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
    
    phi_samples = rbm.get_samples(n_steps=phi_mcmc_steps, warmup=phi_warmup, step=phi_gap, state='h', n=n)

    if init is None:
        # a = rbm.a.copy() + sigma*(np.random.randn(nv) + 1j*np.random.randn(nv))
        # b = rbm.b.copy() + sigma*(np.random.randn(nh) + 1j*np.random.randn(nh))
        # W = rbm.W.copy() + sigma*(np.random.randn(nv, nh) + 1j*np.random.randn(nv, nh))

        ########################################################

        a, b, W = rbm.a.copy(), rbm.b.copy(), rbm.W.copy()

        # l = np.random.rand()
        l = 0.5
        a[n] = l*(a[n] + 1j*np.pi) + (1-l)*(-a[n])
        a += sigma*(np.random.randn(nv) + 1j*np.random.randn(nv)) 

        # l = np.random.rand(nh)
        b = l*b + (1-l)*(b + W[n,:])
        b += sigma*(np.random.randn(nh) + 1j*np.random.randn(nh))

        # l = np.random.rand(nh)
        W[n,:] = l*W[n,:] + (1-l)*(-W[n,:])
        W += sigma*(np.random.randn(nv, nh) + 1j*np.random.randn(nv, nh))

        ########################################################

    else:
        a, b, W = init

    logpsi = RBM(nv, nh)
    logpsi.set_params(a=a, b=b, W=W)
    params = utils.pack_params(a, b, W)

    phiphi = rbm.eval_H(n, phi_samples)
    
    history = []
    F = 0
    F_mean_new = 0.0
    F_mean_old = 0.0
    lr_ = lr
    clock = time()
    t = 0
    
    # while (np.abs(F_mean_new - F_mean_old) > tol or t < 2*lookback + 1) and F_mean_new < 1-tol:
    while (np.abs(F_mean_new - F_mean_old) > tol or t < 2*lookback + 1) and F < 0.99:

        t += 1

        psi_samples = logpsi.get_samples(n_steps=psi_mcmc_steps, warmup=psi_warmup, step=psi_gap)

        phipsi = rbm.eval_H(n, psi_samples)
        psipsi = logpsi(psi_samples)
    
        if fidelity == 'exact': 
            psi_vec = logpsi.get_state_vector(normalized=True)
            F = utils.exact_fidelity(psi_vec, target_state)
        elif fidelity == 'mcmc':
            psiphi = logpsi(phi_samples)
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
        
        if t%resample_phi == 0 and resample_phi is not None:
            phi_samples = rbm.parallel_get_samples(n_steps=phi_mcmc_steps, warmup=phi_warmup, step=phi_gap, state='h', n=n)
            phiphi = rbm.eval_H(n, phi_samples)

        if time() - clock > 20 and verbose:
            print('Iteration {:4d} | Fidelity = {:05.4f} | lr = {:04.3f}'.format(t, F, lr_))
            clock = time()

    return logpsi.a, logpsi.b, logpsi.W, history

def parallel_hadamard_optimization(rbm, comm, n, init=None, tol=1e-6, lookback=50, psi_mcmc_params=(500, 100, 1), phi_mcmc_params=(2000, 500, 1),
                                    sigma=1e-5, resample_phi=None, lr=0.05, lr_tau=None, lr_min=0.0, eps=1e-6, verbose=False):

    r = comm.Get_rank()
    p = comm.Get_size()

    psi_mcmc_steps, psi_warmup, psi_gap = psi_mcmc_params
    phi_mcmc_steps, phi_warmup, phi_gap = phi_mcmc_params
    nv, nh = rbm.nv, rbm.nh
    
    # phi_samples = rbm.parallel_get_samples(comm=comm, n_steps=phi_mcmc_steps, warmup=phi_warmup, step=phi_gap, state='h', n=n) # phi_samples only present on root process, None on rest

    if r==0:
        if init is None:
            a, b, W = rbm.a.copy(), rbm.b.copy(), rbm.W.copy()

            l = 0.5
            a[n] = l*(a[n] + 1j*np.pi) + (1-l)*(-a[n])
            a += sigma*(np.random.randn(nv) + 1j*np.random.randn(nv)) 
            b = l*b + (1-l)*(b + W[n,:])
            b += sigma*(np.random.randn(nh) + 1j*np.random.randn(nh))
            W[n,:] = l*W[n,:] + (1-l)*(-W[n,:])
            W += sigma*(np.random.randn(nv, nh) + 1j*np.random.randn(nv, nh))
        else:
            a, b, W = init
    else:
        a = np.empty(shape=[nv], dtype=np.complex)
        b = np.empty(shape=[nh], dtype=np.complex)
        W = np.empty(shape=[nv, nh], dtype=np.complex)

    comm.Bcast(a, root=0)
    comm.Bcast(b, root=0)
    comm.Bcast(W, root=0)

    logpsi = RBM(nv, nh)
    logpsi.set_params(a=a, b=b, W=W)
    params = utils.pack_params(a, b, W)

    phi_samples = rbm.get_samples(n_steps=phi_mcmc_steps//p, state='h', n=n, warmup=phi_warmup, step=phi_gap)
    phiphi_local = rbm.eval_H(n, phi_samples)
    
    history = []
    F = 0
    F_mean_new = 0.0
    F_mean_old = 0.0
    lr_ = lr
    clock = time()
    t = 0
    
    # while (np.abs(F_mean_new - F_mean_old) > tol or t < 2*lookback + 1) and F_mean_new < 1-tol:
    while (np.abs(F_mean_new - F_mean_old) > tol or t < 2*lookback + 1) and F < 0.99:

        t += 1

        psi_samples = logpsi.get_samples(n_steps=psi_mcmc_steps//p, warmup=psi_warmup, step=psi_gap)

        phipsi_local = rbm.eval_H(n, psi_samples)
        psipsi_local = logpsi(psi_samples)
        psiphi_local = logpsi(phi_samples)

        F_term_1_local = logsumexp(phipsi_local - psipsi_local)
        F_term_2_local = logsumexp(psiphi_local - phiphi_local)

        if r==0:
            F1 = np.empty(shape=[p], dtype=np.complex)
            F2 = np.empty(shape=[p], dtype=np.complex)
        else:
            F1, F2 = None, None

        comm.Gather(F_term_1_local, F1, root=0)
        comm.Gather(F_term_2_local, F2, root=0)

        if r==0:
            logF = logsumexp(F1) + logsumexp(F2) - np.log(phi_mcmc_steps) - np.log(psi_mcmc_steps)
            F = np.exp(logF).real

            history.append(F)

            if t > 2*lookback:
                F_mean_old = F_mean_new
                F_mean_new = sum(history[-lookback:])/lookback

        gas_local, gbs_local, gWs_local = logpsi.grad_log(psi_samples)
        O_local = np.concatenate([gas_local, gbs_local, gWs_local.reshape(-1, nv*nh)], axis=1)

        ratio_psi_local = np.exp(phipsi_local - psipsi_local)

        ratio_psi_mean_local = ratio_psi_local.mean()
        ratio_psi_mean = np.empty(shape=[1], dtype=np.complex)

        comm.Reduce(ratio_psi_mean_local, ratio_psi_mean, root=0)

        if r==0:
            ratio_psi_mean /= p

        comm.Bcast(ratio_psi_mean, root=0)

        grad_logF_local = O_local.mean(axis=0).conj() - (ratio_psi_local.reshape(-1,1)*O_local.conj()).mean(axis=0)/ratio_psi_mean

        if r==0:
            grad_logF= np.empty(shape=params.shape, dtype=np.complex)
        else:
            grad_logF = None

        comm.Reduce(grad_logF_local, grad_logF, root=0)

        if r==0:
            O = np.empty(shape=[p, *O_local.shape], dtype=np.complex)
        else:
            O = None

        comm.Gather(O_local, O, root=0)

        if r==0:
            S = S_matrix(O.reshape(-1, len(params)))
            grad_logF /= p

            grad = F*grad_logF
            delta_theta = solve(S + eps*np.eye(S.shape[0]), grad, assume_a='her')
        
            if lr_tau is not None:
                lr_ = max(lr_min, lr*np.exp(-t/lr_tau))

            params -= lr_*delta_theta
        
        F_mean_old = F_mean_new
        F_mean_new = comm.bcast(F_mean_new, root=0)
        F = comm.bcast(F, root=0)

        comm.Bcast(params, root=0)
        logpsi.a, logpsi.b, logpsi.W = utils.unpack_params(params, nv, nh)
        
        if resample_phi is not None:
            if t%resample_phi == 0:

                phi_samples = rbm.parallel_get_samples(comm=comm, n_steps=phi_mcmc_steps, warmup=phi_warmup, step=phi_gap, state='h', n=n)

                if r==0:
                    phiphi = rbm.eval_H(n, phi_samples)

        if r==0 and time() - clock > 20 and verbose:
            print('Iteration {:4d} | Fidelity = {:05.4f} | lr = {:04.3f}'.format(t, F, lr_))
            clock = time()

    return logpsi.a, logpsi.b, logpsi.W, history