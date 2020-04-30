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

        self.C = 0
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

    def rand_init_weights(self, sigma=0.1):
        self.a = sigma*(np.random.randn(self.nv) + 1j*np.random.randn(self.nv))
        self.b = sigma*(np.random.randn(self.nh) + 1j*np.random.randn(self.nh))
        self.W = sigma*(np.random.randn(self.nv, self.nh) + 1j*np.random.randn(self.nv, self.nh))
        
    def __call__(self, configs, fold_imag=False):

        """
        Evaluates the natural logarithm of the unnormalized wavefunction.
        """
        
        B = np.atleast_2d(configs).astype(np.bool)
        
        term_1 = np.matmul(B, self.a)
        term_2 = utils.log1pexp(self.b.reshape(1,-1) + np.matmul(B, self.W)).sum(axis=1)
        
        logpsi = self.C + term_1 + term_2

        if fold_imag:
            logpsi.imag = (logpsi.imag + np.pi)%(2*np.pi) - np.pi
        
        return logpsi if logpsi.size > 1 else logpsi.item()
            
    def fold_imag_params(self):
        self.C = utils.fold_imag(self.C)
        self.a = utils.fold_imag(self.a)
        self.b = utils.fold_imag(self.b)
        self.W = utils.fold_imag(self.W)
    
    def get_samples(self, n_steps, init=None, state=None, n=None, warmup=0, step=1, verbose=False):
        
        if state is None:
            logp = lambda x: 2*self(x).real
        elif state.lower() == 'h':
            logp = lambda x: 2*self.eval_H(n, x).real
        elif state.lower() == 'x':
            logp = lambda x: 2*self.eval_X(n, x).real
        elif state.lower() == 'z':
            logp = lambda x: 2*self.eval_Z(n, x).real
        else:
            raise KeyError('State {} not recognized.'.format(state))
        
        if init is None:
            previous = np.random.rand(self.nv) < 0.5
        else:
            previous = init

        log_prob_old = logp(previous)
        
        samples = np.zeros(shape=[warmup + step*n_steps, self.nv], dtype=np.bool)
        accept_counter = 0
        
        for t in range(samples.shape[0]):
            
            i = np.random.randint(low=0, high=self.nv)
            
            proposal = previous.copy()
            proposal[i] = not proposal[i]
            
            log_prob_new = logp(proposal)
            
            logA = log_prob_new - log_prob_old
            
            if logA >= np.log(np.random.rand()):
                previous = proposal.copy()
                log_prob_old = log_prob_new
                accept_counter += 1 if t > warmup else 0
                
                samples[t] = proposal
            else:
                samples[t] = previous
        
        if verbose:
            print("Acceptance ratio: ", accept_counter/(n_steps-1))
            
        return samples[warmup::step]
            
    # def parallel_get_samples(self, comm, n_steps, *args, **kwargs):

    #     # This assumes that n_steps is divisible by the number of processes.
        
    #     r = comm.Get_rank()
    #     p = comm.Get_size()
        
    #     l = n_steps//p
    #     extra = n_steps%p

    #     samples = self.get_samples(l, *args, **kwargs)
    #     samples = comm.gather(samples, root=0)

    #     if r == 0:
            
    #         res = np.concatenate(samples, axis=0)

    #         if extra == 0:
    #             return res
    #         else:
    #             extra_samples = self.get_samples(extra, *args, **kwargs)
    #             return np.concatenate([res, extra_samples])
    
    def grad_log(self, configs):
        
        B = np.atleast_2d(configs).astype(np.bool)
        
        ga = configs.copy().astype(np.complex)
        gb = utils.sigmoid(self.b + np.matmul(B, self.W))
        gW = np.matmul(ga[:, :, np.newaxis], gb[:, np.newaxis, :])
        
        return ga, gb, gW
        
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
    
    def eval_X(self, n, configs):
        
        aX = self.a.copy()
        aX[n] = -aX[n]
        bX = self.b + self.W[n,:].copy()
        WX = self.W.copy()
        WX[n,:] = -WX[n,:]
        CX = self.C + self.a[n]
        
        B = configs.reshape([-1, self.nv]).astype(np.bool)
        
        term_1 = np.matmul(B, aX)
        term_2 = utils.log1pexp(bX + np.matmul(B, WX)).sum(axis=1)
        
        return CX + term_1 + term_2
        
    def eval_Z(self, n, configs):
        
        aZ = self.a.copy()
        aZ[n] += 1j*np.pi
        bZ = self.b.copy()
        WZ = self.W.copy()
        CZ = self.C
        
        B = configs.reshape([-1, self.nv]).astype(np.bool)
        
        term_1 = np.matmul(B, aZ)
        term_2 = utils.log1pexp(bZ + np.matmul(B, WZ)).sum(axis=1)
        
        return CZ + term_1 + term_2
    
    def eval_H(self, n, configs):
        xz_vals = np.stack([self.eval_X(n, configs), self.eval_Z(n, configs)], axis=1)
        return logsumexp(xz_vals, axis=1) - np.log(2)/2

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
        self.C += self.a[n] + 1j*np.pi/2

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

    def add_hidden_units(self, num, b_=None, W_=None):

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

        self.num_extra_hs += num
        self.nh += num
        
    def RZZ(self, k, l, phi):

        self.add_hidden_units(num=1)

        B = np.arccosh(np.exp(1j*phi))

        self.W[k,-1] = -2*B
        self.W[l,-1] = 2*B
        self.a[k] += B
        self.a[l] -= B
        self.C += np.log(2)

    def CRZ(self, k, l, phi):

        self.add_hidden_units(num=1)

        A = np.arccosh(np.exp(-1j*phi/2))

        self.W[k,-1] = -2*A
        self.W[l,-1] = 2*A
        self.a[k] += 1j*phi/2 + A
        self.a[l] += 1j*phi/2 - A
        self.C += np.log(2)