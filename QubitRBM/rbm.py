import numpy as np
from scipy import sparse
from scipy.special import logsumexp
import os, sys

libpath = os.path.abspath('..')
if libpath not in sys.path:
    sys.path.append(libpath)

import QubitRBM.utils as utils

class RBM:
    
    def __init__(self, n_visible, n_hidden=1):
        
        self.nv = n_visible
        self.nh = n_hidden
        
        self.W = np.zeros([self.nv, self.nh], dtype=np.complex)
        self.a = np.zeros(self.nv, dtype=np.complex)
        self.b = np.zeros(self.nh, dtype=np.complex)

        self.mask = np.ones(shape=self.W.shape, dtype=np.bool)

        self.C = -self.nh*np.log(2)
        self.num_extra_hs = 0

    def set_params(self, a=None, b=None, W=None):

        if a is not None:
            self.a = a
        if b is not None:
            self.b = b

        if W is not None:
            self.W = W
            self.nv, self.nh = W.shape
        else:
            if a is not None:
                self.nv = len(a)
            if b is not None:
                self.nh = len(b)

    def set_flat_params(self, params):
        self.a = params[:self.nv]
        self.b = params[self.nv:(self.nv + self.nh)]
        self.W[self.mask] = params[(self.nv + self.nh):]

    def get_flat_params(self):
        return np.concatenate([self.a, self.b, self.W[self.mask]], axis=0)

    def num_free_params(self):
        return self.nv + self.nh + self.mask.sum()

    def rand_init_params(self, sigma=0.1, add=False):

        if not add:
            self.a = sigma*(np.random.randn(self.nv) + 1j*np.random.randn(self.nv))
            self.b = sigma*(np.random.randn(self.nh) + 1j*np.random.randn(self.nh))
            self.W[self.mask] = sigma*(np.random.randn(self.mask.sum()) + 1j*np.random.randn(self.mask.sum())) 
        else:
            self.a += sigma*(np.random.randn(self.nv) + 1j*np.random.randn(self.nv))
            self.b += sigma*(np.random.randn(self.nh) + 1j*np.random.randn(self.nh))
            self.W[self.mask] += sigma*(np.random.randn(self.mask.sum()) + 1j*np.random.randn(self.mask.sum())) 
        
    def __call__(self, configs, squeeze=True):

        """
        Evaluates the natural logarithm of the unnormalized wavefunction.
        """
        
        B = np.atleast_2d(configs).astype(np.bool)
        
        term_1 = np.matmul(B, self.a)
        term_2 = utils.log1pexp(self.b.reshape(1,-1) + np.matmul(B, self.W)).sum(axis=1)
        
        logpsi = self.C + term_1 + term_2
        logpsi.imag = (logpsi.imag + np.pi)%(2*np.pi) - np.pi
        
        return logpsi if logpsi.size > 1 else logpsi.item() if squeeze else logpsi
            
    def fold_imag_params(self):
        self.C = utils.fold_imag(self.C)
        self.a = utils.fold_imag(self.a)
        self.b = utils.fold_imag(self.b)
        self.W = utils.fold_imag(self.W)
    
    def get_samples(self, n_steps, state=None, init=None, n_chains=1, warmup=0, step=1, T=1.0, verbose=False, *args, **kwargs):
        
        if init is None:
            previous = np.random.rand(n_chains, self.nv) < 0.5
        else:
            previous = init

        if state is None:
            log_prob_old = 2*self(previous, squeeze=False).real
        elif state.lower() == 'h':
            log_prob_old = np.atleast_1d(2*self.eval_H(previous, *args, **kwargs).real)
        elif state.lower() == 'rx':
            log_prob_old = np.atleast_1d(2*self.eval_RX(previous, *args, **kwargs).real)
        else:
            raise KeyError('Invalid "state": {}'.format(state))
        
        samples = np.zeros(shape=[n_chains, warmup + step*n_steps +1, self.nv], dtype=np.bool)
        samples[:,0,:] = previous.copy()
        accept_counter = np.zeros(n_chains, dtype=np.int)
        
        for t in range(1, samples.shape[1]):

            i = np.random.randint(low=0, high=self.nv, size=n_chains)
            
            proposal = previous.copy()
            proposal[:,i] = np.logical_not(proposal[:,i])
            
            if state is None:
                log_prob_new = 2*self(proposal, squeeze=False).real
            elif state.lower() == 'h':
                log_prob_new = np.atleast_1d(2*self.eval_H(proposal, *args, **kwargs).real)
            elif state.lower() == 'rx':
                log_prob_new = np.atleast_1d(2*self.eval_RX(proposal, *args, **kwargs).real)
            
            logA = (log_prob_new - log_prob_old)/T

            accepted = logA >= np.log(np.random.rand(n_chains))
            not_accepted = np.logical_not(accepted)

            samples[accepted, t, :] = proposal[accepted].copy()
            samples[not_accepted, t, :] = previous[not_accepted].copy()

            previous[accepted] = proposal[accepted].copy()
            log_prob_old[accepted] = log_prob_new[accepted].copy()
            
            if t >= warmup :
                accept_counter += accepted
        
        if verbose:
            print("Mean acceptance ratio: ", np.mean(accept_counter)/(n_steps-1))
            
        return samples[:,(warmup+1)::step,:].reshape(-1, self.nv)

    def get_exact_samples(self, n_samples, state=None, hilbert=None, **kwargs):

        if hilbert is None:
            hilbert = np.array(list(self.hilbert_iter()), dtype=np.bool)

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
    
    def grad_log(self, configs):
        
        B = np.atleast_2d(configs).astype(np.bool)
        
        ga = configs.copy().astype(np.complex)
        gb = utils.sigmoid(self.b + np.matmul(B, self.W))
        gW = np.matmul(ga[:, :, np.newaxis], gb[:, np.newaxis, :])
        
        return np.concatenate([ga, gb, gW[:,self.mask]], axis=1)
        
    def hilbert_iter(self):
        for n in range(2**self.nv):
            yield np.fromiter(map(int, np.binary_repr(n, width=self.nv)), dtype=np.bool, count=self.nv)
            
    def get_state_vector(self, normalized=False):
        """
        If 'normalized', a normalized wavefunction is returned. Otherwise, a vector of complex log-values of the wavefunction is returned.
        """
        
        logpsis = np.fromiter(map(self, self.hilbert_iter()), dtype=np.complex, count=2**self.nv)
        
        if not normalized:
            return logpsis
        else:
            logZ = logsumexp(2*logpsis.real)
            return np.exp(logpsis - 0.5*logZ, dtype=np.complex)
        
    def get_lognorm(self, method='mcmc', samples=None, **mcmc_kwargs):

        if method == 'exact':
            logpsis = np.fromiter(map(self, self.hilbert_iter()), dtype=np.complex, count=2**self.nv)
            return logsumexp(2*logpsis.real)

        elif method == 'mcmc':
            
            if samples is None:
                samples = self.get_samples(**mcmc_kwargs)

            logpsis = self(samples)
            return logsumexp(2*logpsis.real)

        else:
            raise KeyError('Wrong "method". Expected "mcmc" or "exact", got {}'.format(method))
    
    def eval_X(self, configs, n, squeeze=True):
        
        aX = self.a.copy()
        aX[n] = -aX[n]
        bX = self.b + self.W[n,:].copy()
        WX = self.W.copy()
        WX[n,:] = -WX[n,:]
        CX = self.C + self.a[n].copy()
        
        B = np.atleast_2d(configs).astype(np.bool)

        term_1 = np.matmul(B, aX)
        term_2 = utils.log1pexp(bX + np.matmul(B, WX)).sum(axis=1)
        
        res = CX + term_1 + term_2
        res.imag = (res.imag + np.pi)%(2*np.pi) - np.pi 

        return res if res.size > 1 else res.item() if squeeze else res
        
    def eval_Z(self, configs, n, squeeze=True):
        
        aZ = self.a.copy()
        aZ[n] += 1j*np.pi
        bZ = self.b.copy()
        WZ = self.W.copy()
        CZ = self.C
        
        B = np.atleast_2d(configs).astype(np.bool)
        
        term_1 = np.matmul(B, aZ)
        term_2 = utils.log1pexp(bZ + np.matmul(B, WZ)).sum(axis=1)
        
        res = CZ + term_1 + term_2
        res.imag = (res.imag + np.pi)%(2*np.pi) - np.pi 

        return res if res.size > 1 else res.item() if squeeze else res
    
    def eval_H(self, configs, n):
        return logsumexp([self.eval_X(configs, n), self.eval_Z(configs, n)], b=1/np.sqrt(2), axis=0)

    def eval_RX(self, configs, n, beta):
        trig = np.array([np.cos(beta), -1j*np.sin(beta)], dtype=np.complex)
        vals = np.stack([self(configs, squeeze=False), self.eval_X(configs, n, squeeze=False)], axis=1)

        res = logsumexp(vals, b=trig, axis=1)
        res.imag = (res.imag + np.pi)%(2*np.pi) - np.pi
        return res

    def X(self, n):
        """
        Applies the Pauli X gate to qubit n.
        """
        self.a[n] = -self.a[n].copy()
        self.b += self.W[n,:].copy()
        self.W[n,:] = -self.W[n,:].copy()
        self.C += self.a[n].copy()

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
            b_ = np.zeros(shape=[num], dtype=np.complex) 
        if W_ is None:
            W_ = np.zeros(shape=[self.nv, num], dtype=np.complex)

        b = np.zeros(shape=[self.nh+num], dtype=np.complex)
        W = np.zeros(shape=[self.nv, self.nh+num], dtype=np.complex)

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

        self.add_hidden_units(num=1, mask=True)
        self.mask[[k,l], -1] = True

        B = np.arccosh(np.exp(1j*phi))

        self.W[k,-1] = -2*B
        self.W[l,-1] = 2*B
        self.a[k] += B
        self.a[l] -= B
        self.C += np.log(2) - 1j*phi/2

    def CRZ(self, k, l, phi):

        self.add_hidden_units(num=1, mask=True)
        self.mask[[k,l], -1] = True

        A = np.arccosh(np.exp(-1j*phi/2))

        self.W[k,-1] = -2*A
        self.W[l,-1] = 2*A
        self.a[k] += 1j*phi/2 + A
        self.a[l] += 1j*phi/2 - A
        self.C += np.log(2)