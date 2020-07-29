import numpy as np
from scipy.linalg import solve
from scipy.special import logsumexp

from collections import OrderedDict
from time import time
from copy import deepcopy
import os, sys

from numba import njit

libpath = os.path.abspath('..')
if libpath not in sys.path:
    sys.path.append(libpath)

from QubitRBM.rbm import RBM
import QubitRBM.utils as utils

class Optimizer:
    
    def __init__(self, machine, **mcmc_params):
        self.machine = machine
        self.mcmc_params = mcmc_params
        
        self.__printout_template = 'Iteration {:4d} | Fidelity = {:05.4f} | lr = {:04.3f} | Diff mean fidelity = {:08.7f}'
    
    @property
    def machine(self):
        return self.__machine
    
    @machine.setter
    def machine(self, machine):
        assert isinstance(machine, RBM), '"machine" must be an instance of "RBM"'
        self.__machine = machine
        
    @property
    def mcmc_params(self):
        return self.__mcmc_params
    
    @mcmc_params.setter
    def mcmc_params(self, mcmc_params):
        
        keys = ['n_steps', 'n_chains', 'warmup', 'step']
        
        if isinstance(mcmc_params, (dict, OrderedDict)):
            self.__mcmc_params = OrderedDict([(key, val) for key, val in mcmc_params.items() if key in keys])
        else:
            self.__mcmc_params = OrderedDict(zip(keys, mcmc_params))
    
    @staticmethod
    @njit
    def __S_matrix(O):
        T = O.shape[0]
        Oc = O - O.sum(axis=0)/T
        return np.dot(Oc.T.conj(), Oc)/T
    
    @staticmethod
    @njit
    def __grad_F(O, F, psipsi, phipsi):
        ratio_psi = np.exp(phipsi - psipsi)
        ratio_psi_sum = ratio_psi.sum()

        T = O.shape[0]

        grad_logF = O.sum(axis=0).conj()/T - (ratio_psi.reshape(-1,1)*O.conj()).sum(axis=0)/ratio_psi_sum
        return F*grad_logF
    
    @staticmethod
    @njit
    def __grad_log_F(O, psipsi, phipsi):
        ratio_psi = np.exp(phipsi - psipsi)
        ratio_psi_sum = ratio_psi.sum()

        T = O.shape[0]

        grad_logF = O.sum(axis=0).conj()/T - (ratio_psi.reshape(-1,1)*O.conj()).sum(axis=0)/ratio_psi_sum
        return grad_logF
    
    def _sr_update(self, O, psipsi, psiphi, phipsi, phiphi, eps):
        
        F = utils.mcmc_fidelity(psipsi, psiphi, phipsi, phiphi)
        grad = self.__grad_F(O, F, psipsi, phipsi)
        
        S = self.__S_matrix(O)
        S[np.diag_indices_from(S)] += eps 
        
        delta_theta = solve(S, grad, overwrite_a=True, overwrite_b=True, assume_a='her')
        
        return F, delta_theta

    @staticmethod
    def maxcut_cost(graph, configs):
        B = np.atleast_2d(configs)
        z = (-1)**B[:, graph.edges()]
        return z.prod(axis=-1).sum(axis=-1)

    @staticmethod
    def grad_maxcut_cost(Os, costs):
        term_1 = (costs.reshape(-1,1)*Os.conj()).mean(axis=0)
        term_2 = costs.mean()*Os.conj().mean(axis=0)
        return term_1 - term_2
    
    def sr_rx(self, n, beta, tol=1e-3, lookback=5, max_iters=1000, resample_phi=None,
              lr=1e-1, lr_tau=None, lr_min=0.0, eps=1e-4, verbose=False):
        
        nv, nh = self.machine.nv, self.machine.nh

        logpsi = deepcopy(self.machine)

        if np.abs(np.sin(beta)) > np.abs(np.cos(beta)):
            logpsi.X(n)

        params = logpsi.params
        phi_samples = self.machine.get_samples(**self.mcmc_params, state='rx', n=n, beta=beta)
        phiphi = self.machine.eval_RX(phi_samples, n=n, beta=beta)

        history = []
        F, F_mean_new, F_mean_old = 0.0, 0.0, 0.0
        diff_mean_F = 2*tol
        lr_ = lr
        clock = time()
        t = 0

        while (diff_mean_F > tol or t < 2*lookback + 1) and F_mean_new < 1-tol and t < max_iters:

            t += 1

            psi_samples = logpsi.get_samples(**self.mcmc_params)

            psipsi = logpsi(psi_samples)
            phipsi = self.machine.eval_RX(psi_samples, n=n, beta=beta)
            psiphi = logpsi(phi_samples)

            O = logpsi.grad_log(psi_samples)
            F, delta_theta = self._sr_update(O, psipsi, psiphi, phipsi, phiphi, eps)

            params -= lr_*delta_theta
            logpsi.params = params
            
            history.append(F)
            
            if lr_tau is not None:
                lr_ = max(lr_min, lr*np.exp(-t/lr_tau))
            
            if t > 2*lookback:
                F_mean_old = sum(history[-2*lookback:-lookback])/lookback
                F_mean_new = sum(history[-lookback:])/lookback
                
            diff_mean_F = np.abs(F_mean_new - F_mean_old)

            if resample_phi is not None:
                if t%resample_phi == 0:
                    phi_samples = self.machine.get_samples(**self.mcmc_params, state='rx', beta=beta, n=n)
                    phiphi = self.machine.eval_RX(phi_samples, n=n, beta=beta)
            
            if time() - clock > 10 and verbose:
                print(self.__printout_template.format(t, F, lr_, diff_mean_F))
                clock = time()

        return params, np.asarray(history)
    
    def sr_compress(self, init, tol=1e-3, lookback=5, max_iters=1000, resample_phi=None,
                    lr=1e-1, lr_tau=None, lr_min=0.0, eps=1e-4, verbose=False):
        
        nv = self.machine.nv
        target_nh = int((len(init) - nv)/(nv + 1))

        logpsi = RBM(n_visible=nv, n_hidden=target_nh)
        logpsi.params = init

        params = init.copy()
        phi_samples = self.machine.get_samples(**self.mcmc_params)
        phiphi = self.machine(phi_samples)

        history = []
        F, F_mean_new, F_mean_old = 0.0, 0.0, 0.0
        diff_mean_F = 2*tol
        lr_ = lr
        clock = time()
        t = 0

        while (diff_mean_F > tol or t < 2*lookback + 1) and F_mean_new < 1-tol and t < max_iters:

            t += 1

            psi_samples = logpsi.get_samples(**self.mcmc_params)

            psipsi = logpsi(psi_samples)
            phipsi = self.machine(psi_samples)
            psiphi = logpsi(phi_samples)
            
            O = logpsi.grad_log(psi_samples)
            F, delta_theta = self._sr_update(O, psipsi, psiphi, phipsi, phiphi, eps)

            params -= lr_*delta_theta
            logpsi.params = params
            
            history.append(F)
            
            if lr_tau is not None:
                lr_ = max(lr_min, lr*np.exp(-t/lr_tau))

            if t > 2*lookback:
                F_mean_old = sum(history[-2*lookback:-lookback])/lookback
                F_mean_new = sum(history[-lookback:])/lookback
                
            diff_mean_F = np.abs(F_mean_new - F_mean_old)

            if resample_phi is not None:
                if t%resample_phi == 0:
                    phi_samples = self.machine.get_samples(**self.mcmc_params)
                    phiphi = self.machine(phi_samples)

            if time() - clock > 10 and verbose:
                print(self.__printout_template.format(t, F, lr_, diff_mean_F))
                clock = time()

        return logpsi, np.asarray(history)
    
    def adam_compress(self, rbm, init, lr=1e-3, tol=1e-3, lookback=20, resample_phi=None, max_iters=1000,
                     betas=(0.9, 0.999), eps=1e-6, verbose=False):
    
        nv = self.machine.nv
        target_nh = int((len(init) - nv)/(nv + 1))

        logpsi = RBM(n_visible=nv, n_hidden=target_nh)
        logpsi.params = init

        params = init.copy()
        phi_samples = self.machine.get_samples(**self.mcmc_params)
        phiphi = self.machine(phi_samples)

        history = []
        F, F_mean_new, F_mean_old = 0.0, 0.0, 0.0
        diff_mean_F = 2*tol
        lr_ = lr
        clock = time()
        t = 0

        beta1, beta2 = betas
        mr = np.zeros(len(params))
        mi = np.zeros(len(params))
        vr = np.zeros(len(params))
        vi = np.zeros(len(params))

        while (diff_mean_F > tol or t < 2*lookback + 1) and F_mean_new < 0.999 and t < max_iters:

            t += 1

            psi_samples = logpsi.get_samples(**self.mcmc_params)
            
            psipsi = logpsi(psi_samples)
            phipsi = self.machine(psi_samples)

            O = logpsi.grad_log(psi_samples)
            grad_logF = self.__grad_log_F(O, psipsi, phipsi)

            ########## Adam update ##########

            gr, gi = np.real(grad_logF), np.imag(grad_logF)

            mr = beta1*mr + (1-beta1)*gr
            mi = beta1*mi + (1-beta1)*gi
            vr = beta2*vr + (1-beta2)*(gr**2)
            vi = beta2*vi + (1-beta2)*(gi**2)

            mr_hat = mr/(1 - beta1**t)
            mi_hat = mi/(1 - beta1**t)
            vr_hat = vr/(1 - beta2**t)
            vi_hat = vi/(1 - beta2**t)

            d_params = mr_hat/(np.sqrt(vr_hat) + eps) + 1j*(mi_hat/(np.sqrt(vi_hat) + eps))

            ################################

            params -= lr*d_params
            logpsi.params = params
            
            
            psiphi = logpsi(phi_samples)
            F = utils.mcmc_fidelity(psipsi, psiphi, phipsi, phiphi)
            history.append(F)

            if t > 2*lookback:
                F_mean_old = sum(history[-2*lookback:-lookback])/lookback
                F_mean_new = sum(history[-lookback:])/lookback
                
            diff_mean_F = np.abs(F_mean_new - F_mean_old)

            if resample_phi is not None:
                if t%resample_phi == 0:
                    phi_samples = rbm.get_samples(**self.mcmc_params)
                    phiphi = self.machine(phi_samples)

            if time() - clock > 20 and verbose:
                diff_mean_F = np.abs(F_mean_new - F_mean_old)
                print(self.__printout_template.format(t, F, lr, diff_mean_F))
                clock = time()

        return logpsi, np.asarray(history)

    def direct_sr(self, graph, tol=5e-2, lookback=5, max_iters=1000, lr=5e-2, eps=1e-4, verbose=False):

        nv, nh = self.machine.nv, self.machine.nh

        logpsi = deepcopy(self.machine)
        params = logpsi.params

        history = []
        C, C_mean_new, C_mean_old = 0.0, 0.0, 0.0
        diff_mean_C = 2*tol
        clock = time()
        t = 0

        while (diff_mean_C > tol or t < 2*lookback + 1) and C_mean_new < 1-tol and t < max_iters:
            
            t += 1
            
            psi_samples = logpsi.get_samples(**self.mcmc_params)
            
            costs = self.maxcut_cost(graph, psi_samples)
            Os = logpsi.grad_log(psi_samples)
            
            grad = self.grad_maxcut_cost(Os, costs)
            S = self.__S_matrix(Os)
            S[np.diag_indices_from(S)] += eps
            
            d_params = solve(S, grad, assume_a='her')
            
            params -= lr*d_params
            logpsi.params = params
            
            C = costs.mean()
            history.append(C)
            
            if t > 2*lookback:
                C_mean_old = sum(history[-2*lookback:-lookback])/lookback
                C_mean_new = sum(history[-lookback:])/lookback
                
            diff_mean_C = np.abs(C_mean_new - C_mean_old)
            
            if time() - clock > 10 and verbose:
                print(self.__printout_template.format(t, C, lr, diff_mean_C))
                clock = time()
                
        return params, np.asarray(history)