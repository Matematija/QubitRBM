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

from mpi4py import MPI

def S_matrix(O):
    O_centered = O - O.mean(axis=0, keepdims=True)
    return np.matmul(O_centered.T.conj(), O_centered)/O.shape[0]

def rx_optimization(rbm, n, beta, tol=1e-6, lookback=50, max_iters=10000, psi_mcmc_params=(500,5,50,1), phi_mcmc_params=(500,5,50,1),
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

    while (np.abs(F_mean_new - F_mean_old) > tol or t < 2*lookback + 1) and F_mean_new < 0.999 and t < max_iters:
        
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
        delta_theta = solve(S, grad, overwrite_a=True, overwrite_b=True, assume_a='her')
        
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

def parallel_rx_optimization(comm, rbm, n, beta, tol=1e-6, lookback=50, max_iters=10000,
                            proc_psi_mcmc_params=(500,5,50,1), proc_phi_mcmc_params=(500,5,50,1),
                            resample_phi=None, lr=5e-2, lr_tau=None, lr_min=0.0, eps=1e-6, verbose=False):

    r = comm.Get_rank()
    p = comm.Get_size()

    psi_mcmc_args = dict(zip(['n_steps', 'n_chains', 'warmup', 'step'], proc_psi_mcmc_params))
    phi_mcmc_args = dict(zip(['n_steps', 'n_chains', 'warmup', 'step'], proc_phi_mcmc_params))
    nv, nh = rbm.nv, rbm.nh

    logpsi = deepcopy(rbm)
    
    if np.abs(np.sin(beta)) > np.abs(np.cos(beta)):
        logpsi.X(n)
    
    logpsi.fold_imag_params()
    params = logpsi.get_flat_params()

    # print('Process {} sampling phi'.format(r))

    phi_samples = rbm.get_samples(**phi_mcmc_args, state='rx', n=n, beta=beta)
    phiphi_local = rbm.eval_RX(phi_samples, n=n, beta=beta)

    # print('Process {} done sampling phi'.format(r))

    F = 0
    t = 0
    loop = True
    clock = time()

    if r==0:
        history = []
        F_mean_new = 0.0
        F_mean_old = 0.0
        lr_ = lr
    
    while loop:
        
        t += 1

        # print('Process {} sampling psi'.format(r))

        psi_samples = logpsi.get_samples(**psi_mcmc_args)

        # print('Process {} evaluating fns'.format(r))
        
        psipsi_local = logpsi(psi_samples)
        phipsi_local = rbm.eval_RX(psi_samples, n=n, beta=beta)
        psiphi_local = logpsi(phi_samples)

        ratio_psi_local = np.exp(phipsi_local - psipsi_local)

        # Calculating the fidelity:

        # print('Process {} evaluating fidelity'.format(r))

        F_fac_1_local = ratio_psi_local.mean()
        F_fac_2_local = np.exp(psiphi_local - phiphi_local).mean()

        if r==0:
            F_fac_1 = np.empty(shape=[1], dtype=np.complex)
            F_fac_2 = np.empty(shape=[1], dtype=np.complex)
        else:
            F_fac_1, F_fac_2 = None, None

        # print('Process {} reducing fidelity'.format(r))

        comm.Reduce(F_fac_1_local, F_fac_1, op=MPI.SUM, root=0)
        comm.Reduce(F_fac_2_local, F_fac_2, op=MPI.SUM, root=0)
        
        if r==0:
            F = np.real(F_fac_1*F_fac_2).item()/p**2
            history.append(F)

        # Calculating gradient of log F

        # print('Process {} calculating grad log F'.format(r))
        
        ratio_psi_mean_local = ratio_psi_local.mean()

        ratio_psi_mean = comm.allreduce(ratio_psi_mean_local)
        ratio_psi_mean /= p

        O_local = logpsi.grad_log(psi_samples)

        grad_logF_local = O_local.mean(axis=0).conj() - (ratio_psi_local.reshape(-1,1)*O_local.conj()).mean(axis=0)/ratio_psi_mean

        if r==0:
            grad_logF= np.empty(shape=params.shape, dtype=np.complex)
        else:
            grad_logF = None

        # print('Process {} reducing grad log F'.format(r))

        comm.Reduce(grad_logF_local, grad_logF, root=0)

        # print('Process {} reducing O'.format(r))

        if r==0:
            O = np.empty(shape=[p, *O_local.shape], dtype=np.complex)
        else:
            O = None

        comm.Gather(O_local, O, root=0)

        if r==0:
            grad_logF /= p
            grad = F*grad_logF

            # print('Process {} calculating S'.format(r))

            S = S_matrix(O.reshape(-1, len(params)))

            # print('Process {} solving for delta'.format(r))

            S[np.diag_indices_from(S)] += eps 
            delta_theta = solve(S, grad, overwrite_a=True, overwrite_b=True, assume_a='her')
        
            if lr_tau is not None:
                lr_ = max(lr_min, lr*np.exp(-t/lr_tau))

            params -= lr_*delta_theta

        # print('Process {} broadcasting params'.format(r))

        comm.Bcast(params, root=0)
        logpsi.set_flat_params(params)

        # print('Process {} resampling phi'.format(r))

        if resample_phi is not None:
            if t%resample_phi == 0:
                phi_samples = rbm.get_samples(**phi_mcmc_args, state='rx', beta=beta, n=n)
                phiphi_local = rbm.eval_RX(phi_samples, n=n, beta=beta)

        if r==0:
            if t > 2*lookback:
                F_mean_old = sum(history[-2*lookback:-lookback])/lookback
                F_mean_new = sum(history[-lookback:])/lookback

            loop = (np.abs(F_mean_new - F_mean_old) > tol or t < 2*lookback + 1) and F_mean_new < 0.999 and t < max_iters

        loop = comm.bcast(loop, root=0)

        if time() - clock > 5 and verbose and r==0:
            diff_mean_F = np.abs(F_mean_new - F_mean_old)
            print('Iteration {:4d} | Fidelity = {:05.4f} | lr = {:04.3f} | diff_mean_F = {:08.7f}'.format(t, F, lr_, diff_mean_F))
            clock = time()

    if r==0:
        return params, history

def compress_rbm(rbm, target_hidden_num, init, tol=1e-6, lookback=50, max_iters=10000, psi_mcmc_params=(500,5,50,1), phi_mcmc_params=(500,5,50,1),
                    sigma=1e-5, resample_phi=None, lr=5e-2, lr_tau=None, lr_min=0.0, eps=1e-6, verbose=False):

    psi_mcmc_args = dict(zip(['n_steps', 'n_chains', 'warmup', 'step'], psi_mcmc_params))
    phi_mcmc_args = dict(zip(['n_steps', 'n_chains', 'warmup', 'step'], phi_mcmc_params))
    nv, nh = rbm.nv, rbm.nh

    logpsi = RBM(n_visible=nv, n_hidden=target_hidden_num)
    logpsi.set_flat_params(init)

    params = init.copy()
    phi_samples = rbm.get_samples(**phi_mcmc_args)
    
    phiphi = rbm(phi_samples)
    
    history = []
    F = 0
    F_mean_new = 0.0
    F_mean_old = 0.0
    lr_ = lr
    clock = time()
    t = 0

    while (np.abs(F_mean_new - F_mean_old) > tol or t < 2*lookback + 1) and F_mean_new < 0.999 and t < max_iters:
        
        t += 1

        psi_samples = logpsi.get_samples(**psi_mcmc_args)
        
        psipsi = logpsi(psi_samples)
        phipsi = rbm(psi_samples)
        psiphi = logpsi(phi_samples)
        
        F = utils.mcmc_fidelity(psipsi, psiphi, phipsi, phiphi)
        history.append(F)

        O = logpsi.grad_log(psi_samples)
        S = S_matrix(O)

        ratio_psi = np.exp(phipsi - psipsi)
        ratio_psi_mean = ratio_psi.mean()

        grad_logF = O.mean(axis=0).conj() - (ratio_psi.reshape(-1,1)*O.conj()).mean(axis=0)/ratio_psi_mean
        grad = F*grad_logF

        S[np.diag_indices_from(S)] += eps 
        delta_theta = solve(S, grad, overwrite_a=True, overwrite_b=True, assume_a='her')
        
        if lr_tau is not None:
            lr_ = max(lr_min, lr*np.exp(-t/lr_tau))

        params -= lr_*delta_theta
        logpsi.set_flat_params(params)

        if t > 2*lookback:
            F_mean_old = sum(history[-2*lookback:-lookback])/lookback
            F_mean_new = sum(history[-lookback:])/lookback
        
        if resample_phi is not None:
            if t%resample_phi == 0:
                phi_samples = rbm.get_samples(**phi_mcmc_args)
                phiphi = rbm(phi_samples)

        if time() - clock > 5 and verbose:
            diff_mean_F = np.abs(F_mean_new - F_mean_old)
            print('Iteration {:4d} | Fidelity = {:05.4f} | lr = {:04.3f} | diff_mean_F = {:08.7f}'.format(t, F, lr_, diff_mean_F))
            clock = time()

    return logpsi, history

def qaoadam(qaoa, lr, tol, init=None, betas=(0.9, 0.999), eps=1e-6, hilbert=None, dx=1e-5, verbose=True):

    beta1, beta2 = betas
    
    if init is None:
        gamma = np.random.uniform(0, np.pi/2, size=qaoa.p)
        beta = np.random.uniform(-np.pi/4, np.pi/4, size=qaoa.p)
        params = np.concatenate([gamma, beta])
    else:
        params = np.array(init).copy()

    if hilbert is None:
        hilbert = np.array(list(utils.hilbert_iter(qaoa.n_qubits)))

    m = np.zeros_like(params)
    v = np.zeros_like(params)
    t = 0
    clock = time()
    history = []

    while True:

        t += 1
        
        g, b = np.split(params, 2)
        grad = qaoa.grad_cost(g, b, hilbert=hilbert, dx=dx)

        m = beta1*m + (1-beta1)*grad
        v = beta2*v + (1-beta2)*grad**2

        m_hat = m/(1 + beta1**t)
        v_hat = v/(1 + beta2**t)

        d_params = lr * m_hat/(np.sqrt(v_hat) + eps)

        if np.any(d_params > tol):
            params -= d_params
        else:
            break

        f = qaoa.cost_from_params(g, b, hilbert=hilbert)
        history.append(f)

        if time() - clock > 10 and verbose:
            print('Iteration {} | Cost = {}'.format(t, f))
            clock = time()

    return params, np.array(history)