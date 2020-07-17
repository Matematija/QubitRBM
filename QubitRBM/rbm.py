import numpy as np
from scipy.special import logsumexp
from collections import OrderedDict

from numba import njit
import os, sys

libpath = os.path.abspath('..')
if libpath not in sys.path:
    sys.path.append(libpath)

import QubitRBM.utils as utils
from QubitRBM.utils import log1pexp, logaddexp

@njit
def _eval_RBM_from_params(B, C, a, b, W):
    return C + np.dot(B, a) + log1pexp(b + np.dot(B, W)).sum(axis=1)

@njit
def _eval_RX_RBM_from_params(B, C, a, b, W, CX, aX, bX, WX, beta):

    vals = _eval_RBM_from_params(B, C, a, b, W)
    Xvals = _eval_RBM_from_params(B, CX, aX, bX, WX)

    trig = (np.cos(beta), -1j*np.sin(beta))

    return logaddexp(vals, Xvals, b=trig)

########################################################################

class RBM:
    
    def __init__(self, n_visible, n_hidden=1):
        
        self.nv = n_visible
        self.nh = n_hidden
        
        self.W = np.zeros([self.nv, self.nh], dtype=complex)
        self.a = np.zeros(self.nv, dtype=complex)
        self.b = np.zeros(self.nh, dtype=complex)

        self.mask = np.ones(shape=self.W.shape, dtype=np.bool)

        self.C = -(self.nh + self.nv/2)*np.log(2)
        self.num_extra_hs = 0

        self.__eval_from_params = _eval_RBM_from_params
        self.__eval_RX_from_params = _eval_RX_RBM_from_params

    @property
    def params(self):
        return np.concatenate([self.a, self.b, self.W[self.mask]], axis=0)

    @params.setter
    def params(self, params):
        self.a = params[:self.nv].copy()
        self.b = params[self.nv:(self.nv + self.nh)].copy()
        self.W[self.mask] = params[(self.nv + self.nh):].copy()

    @property
    def state_dict(self):
        return OrderedDict(zip(['a', 'b', 'W'], [self.a, self.b. self.W]))

    @property
    def num_free_params(self):
        return self.nv + self.nh + self.mask.sum()

    @property
    def alpha(self):
        return self.nh/self.nv

    def set_params(self, C=None, a=None, b=None, W=None, mask=None):

        if C is not None:
            self.C = C
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b

        if W is not None:
            self.nv, self.nh = W.shape
            self.W = W
        else:
            if a is not None:
                self.nv = len(a)
            if b is not None:
                self.nh = len(b)

        if mask is not None:
            self.mask = mask
        if self.mask.shape != self.W.shape:
            self.mask = np.ones_like(self.W, dtype=np.bool)

    def rand_init_params(self, sigma=0.1, add=False):
        noise = sigma*(np.random.randn(self.num_free_params) + 1j*np.random.randn(self.num_free_params)) 
        self.params = self.params + noise if not add else noise
        
    def __call__(self, configs, fold=True, squeeze=True):

        """
        Evaluates the natural logarithm of the unnormalized wavefunction.
        """
        
        B = np.atleast_2d(configs).astype(complex)
        logpsi = self.__eval_from_params(B, self.C, self.a, self.b, self.W)

        if fold:
            logpsi.imag = (logpsi.imag + np.pi)%(2*np.pi) - np.pi
        
        # return logpsi if logpsi.size > 1 else logpsi.item() if squeeze else logpsi
        return logpsi
            
    def fold_imag_params(self):
        self.C = utils.fold_imag(self.C)
        self.a = utils.fold_imag(self.a)
        self.b = utils.fold_imag(self.b)
        self.W = utils.fold_imag(self.W)

    def iter_samples(self, n_steps, state=None, init=None, n_chains=1, warmup=0, step=1, T=1.0, verbose=False, n=None, beta=None):
        
        if init is None:
            init_ = (np.random.rand(n_chains, self.nv) < 0.5).astype(complex)
        else:
            init_ = np.atleast_2d(init).astype(complex)

        if state is None:
            return self.__numba_sample_iter(self.C, self.a, self.b, self.W, n_steps, init_, n_chains, warmup, step, T)
        elif state.lower() == 'rx':
            return self.__numba_X_sample_iter(self.C, self.a, self.b, self.W, n, beta, n_steps, init_, n_chains, warmup, step, T)
        else:
            raise KeyError('Invalid "state": {}'.format(state))

    @staticmethod
    @njit
    def __numba_sample_iter(C, a, b, W, n_steps, init_, n_chains=1, warmup=0, step=1, T=1):

        samples = init_
        log_prob_old = 2*np.real(_eval_RBM_from_params(init_, C, a, b, W))
        nv = init_.shape[-1]

        accept_counter = np.zeros(n_chains)
        
        for t in range(warmup + step*n_steps):

            i = np.random.randint(low=0, high=nv, size=n_chains)
            
            proposal = samples.copy()
            proposal[:,i] = 1.0 - proposal[:,i]
            
            log_prob_new = 2*np.real(_eval_RBM_from_params(proposal, C, a, b, W))
            
            logA = (log_prob_new - log_prob_old)/T
            accepted = logA >= np.log(np.random.rand(n_chains))

            samples[accepted,:] = proposal[accepted,:]

            accept_counter += accepted

            if t >= warmup and (t-warmup)%step == 0:
                yield samples.real.copy()

            log_prob_old[accepted] = log_prob_new[accepted].copy()

    @staticmethod
    @njit
    def __numba_X_sample_iter(C, a, b, W, n, beta, n_steps, init_, n_chains=1, warmup=0, step=1, T=1):

        aX = a.copy()
        aX[n] = -aX[n]
        bX = b + W[n,:]
        WX = W.copy()
        WX[n,:] = -WX[n,:]
        CX = C + a[n]

        samples = init_
        log_prob_old = 2*np.real(_eval_RX_RBM_from_params(init_, C, a, b, W, CX, aX, bX, WX, beta))
        nv = init_.shape[-1]

        accept_counter = np.zeros(n_chains)
        
        for t in range(warmup + step*n_steps):

            i = np.random.randint(low=0, high=nv, size=n_chains)
            
            proposal = samples.copy()
            proposal[:,i] = 1.0 - proposal[:,i]
            
            log_prob_new = 2*np.real(_eval_RX_RBM_from_params(proposal, C, a, b, W, CX, aX, bX, WX, beta))
            
            logA = (log_prob_new - log_prob_old)/T
            accepted = logA >= np.log(np.random.rand(n_chains))

            samples[accepted,:] = proposal[accepted,:]

            accept_counter += accepted

            if t >= warmup and (t-warmup)%step == 0:
                yield samples.real.copy()

            log_prob_old[accepted] = log_prob_new[accepted].copy()

    def get_samples(self, *args, **kwargs):
        samples = np.array(list(self.iter_samples(*args, **kwargs)))
        return samples.reshape(-1, self.nv)

    def get_exact_samples(self, n_samples, state=None, hilbert=None, **kwargs):

        if hilbert is None:
            hilbert = np.array(list(utils.hilbert_iter(self.nv)), dtype=np.bool)

        if state is None:
            logvals = self(hilbert)
        elif state.lower() == 'rx':
            logvals = self.eval_RX(hilbert, **kwargs)
        elif state.lower() == 'h':
            logvals = self.eval_H(hilbert, **kwargs)
        else:
            raise KeyError('Invalid "state": {}'.format(state))
        
        p = np.exp(2*logvals.real - logsumexp(2*logvals.real))
        inds = np.random.choice(np.arange(2**self.nv), size=n_samples, p=p)

        return hilbert[inds]

    @staticmethod
    @njit
    def __grad_log_from_params(B, a, b, W):
        ga = B.copy()
        gb = utils.sigmoid(b + np.dot(B, W))
        gW = np.expand_dims(ga, 2)*np.expand_dims(gb, 1)
        return ga, gb, gW

    def grad_log(self, configs):
        B = np.atleast_2d(configs).astype(complex)
        ga, gb, gW = self.__grad_log_from_params(B, self.a, self.b, self.W)
        return np.concatenate([ga, gb, gW[:,self.mask]], axis=1)
            
    def get_state_vector(self, normalized=False, state=None, hilbert=None, **kwargs):
        """
        If 'normalized', a normalized wavefunction is returned. Otherwise, a vector of complex log-values of the wavefunction is returned.
        """
        
        if hilbert is None:
            hilbert = np.array(list(utils.hilbert_iter(self.nv))) 
        
        if state is None:
            logvals = self(hilbert)
        elif state.lower() == 'rx':
            logvals = self.eval_RX(hilbert, **kwargs)
        elif state.lower() == 'h':
            logvals = self.eval_H(hilbert, **kwargs)
        else:
            raise KeyError('Invalid "state": {}'.format(state)) 
        
        if not normalized:
            return logvals
        else:
            logZ = logsumexp(2*logvals.real)
            return np.exp(logvals - 0.5*logZ)

    def __get_X_params(self, n):

        aX = self.a.copy()
        aX[n] = -aX[n]
        bX = self.b + self.W[n,:].copy()
        WX = self.W.copy()
        WX[n,:] = -WX[n,:]
        CX = self.C + self.a[n].copy()

        return CX, aX, bX, WX
    
    def eval_X(self, configs, n, fold=True, squeeze=True):
        
        CX, aX, bX, WX = self.__get_X_params(n)

        B = np.atleast_2d(configs).astype(complex)
        res = self.__eval_from_params(B, CX, aX, bX, WX)

        if fold:
            return utils.fold_imag(res)
        else:
            return res
        
    def eval_Z(self, configs, n, fold=True, squeeze=True):
        
        aZ = self.a.copy()
        aZ[n] += 1j*np.pi
        bZ = self.b.copy()
        WZ = self.W.copy()
        CZ = self.C

        B = np.atleast_2d(configs).astype(complex)
        res = self.__eval_from_params(B, CZ, aZ, bZ, WZ)
        
        if fold:
            return utils.fold_imag(res)
        else:
            res
    
    def eval_H(self, configs, n, fold=True):

        Xvals = self.eval_X(configs, n=n, fold=False, squeeze=False)
        Zvals = self.eval_Z(configs, n=n, fold=False, squeeze=False)

        b = (1/np.sqrt(2), 1/np.sqrt(2))
        res = logaddexp(Xvals, Zvals, b=b)

        if fold:
            return utils.fold_imag(res)
        else:
            return res

    def eval_RX(self, configs, n, beta, fold=True):

        B = np.atleast_2d(configs).astype(complex)

        C, a, b, W = self.C, self.a, self.b, self.W
        CX, aX, bX, WX = self.__get_X_params(n)

        res = self.__eval_RX_from_params(B, C, a, b, W, CX, aX, bX, WX, beta)

        if fold:
            res.imag = (res.imag + np.pi)%(2*np.pi) - np.pi

        return res

    def X(self, n):
        """
        Applies the Pauli X gate to qubit n.
        """
        self.a[n] = -self.a[n].copy()
        self.b += self.W[n,:].copy()
        self.W[n,:] = -self.W[n,:].copy()
        self.C -= self.a[n].copy()

    def Y(self, n):
        """
        Applies the Pauli Y gate to qubit n.
        """
        self.a[n] = -self.a[n].copy() + 1j*np.pi
        self.b += self.W[n,:].copy()
        self.W[n,:] = -self.W[n,:].copy()
        self.C += self.a[n].copy() + 1j*np.pi/2

    def Z(self, n):
        """
        Applies the Pauli Z gate to qubit n.
        """
        self.a[n] += 1j*np.pi

    def RZ(self, n, phi):
        """
        Applies the following unitary gate to qubit n: e^{-i Z_n \phi /2} ~ [[1, 0], [0, e^{i \phi}]] .
        """
        self.a[n] += 1j*phi

    def P(self, n, phi):
        self.X(n)
        self.RZ(n, phi)

    def add_hidden_units(self, num, b_=None, W_=None, mask=False):

        if b_ is None: 
            b_ = np.zeros(shape=[num], dtype=complex) 
        if W_ is None:
            W_ = np.zeros(shape=[self.nv, num], dtype=complex)

        b = np.zeros(shape=[self.nh+num], dtype=complex)
        W = np.zeros(shape=[self.nv, self.nh+num], dtype=complex)

        b[:-num] = self.b
        b[-num:] = b_
        self.b = b

        W[:,:-num] = self.W
        W[:,-num:] = W_
        self.W = W
        self.C -= np.log(2)

        self.num_extra_hs += num
        self.nh += num

        if mask:
            m = np.zeros(shape=[self.nv, num], dtype=np.bool)
        else:
            m = np.ones(shape=[self.nv, num], dtype=np.bool)

        self.mask = np.concatenate([self.mask, m], axis=1)
        
    def RZZ(self, k, l, phi):

        self.add_hidden_units(num=1, mask=False)
        self.mask[[k,l], -1] = True

        B = np.arccosh(np.exp(1j*phi))

        self.W[k,-1] = -2*B
        self.W[l,-1] = 2*B
        self.a[k] += B
        self.a[l] -= B
        # self.C += np.log(2) - 1j*phi/2
        self.C -= np.log(2)

    def CRZ(self, k, l, phi):

        self.add_hidden_units(num=1, mask=False)
        self.mask[[k,l], -1] = True

        A = np.arccosh(np.exp(-1j*phi/2))

        self.W[k,-1] = -2*A
        self.W[l,-1] = 2*A
        self.a[k] += 1j*phi/2 + A
        self.a[l] += 1j*phi/2 - A
        self.C -= np.log(2)

    def save(self, path, **kwargs):
        np.savez(path, C=self.C, a=self.a, b=self.b, W=self.W, mask=self.mask, **kwargs)

    def load(self, path):

        f = np.load(path)

        C, a, b, W, mask = f['C'].copy(), f['a'].copy(), f['b'].copy(), f['W'].copy(), f['mask'].copy()
        self.set_params(C=C, a=a, b=b, W=W, mask=mask)

        return f