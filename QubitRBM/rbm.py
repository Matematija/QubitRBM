import numpy as np
from scipy.special import logsumexp
from scipy import sparse

import os, sys
libpath = os.path.abspath('..')
if libpath not in sys.path:
    sys.path.append(libpath)

try:
    import QubitRBM.utils as utils
except Exception as error:
    print('QubitRBM folder not in PATH!')
    raise(error)

class RBM:
    
    def __init__(self, n_visible, n_hidden):
        
        self.nv = n_visible
        self.nh = n_hidden
        
        self.W = np.zeros([self.nv, self.nh], dtype=np.complex)
        self.a = np.zeros(self.nv, dtype=np.complex)
        self.b = np.zeros(self.nh, dtype=np.complex)
        
        self.C = 0

        max_extra_hs = self.nv*(self.nv - 1)//2
        # self.W_ = sparse.dok_matrix((self.nv, max_extra_hs), dtype=np.complex)
        self.W_ = None
        self.extra_hs = []

    def set_params(self, a, b, W):

        assert len(a) == self.nv, 'Wrong visible bias array length! Expected {}, got {}.'.format(self.nv, len(a))
        assert len(b) == self.nh, 'Wrong hidden bias array length! Expected {}, got {}.'.format(self.nh, len(b))
        assert W.shape == self.W.shape, 'Wrong weights shape! Expected {}, got {}.'.format(self.W.shape, W.shape)

        self.a = a
        self.b = b
        self.W = W

    def rand_init_weights(self, sigma=0.1):
        self.a = sigma*(np.random.randn(self.nv) + 1j*np.random.randn(self.nv))
        self.b = sigma*(np.random.randn(self.nh) + 1j*np.random.randn(self.nh))
        self.W = sigma*(np.random.randn(self.nv, self.nh) + 1j*np.random.randn(self.nv, self.nh))
        
    def __call__(self, configs):

        """
        Evaluates the natural logarithm of the unnormalized wavefunction.
        """
        
        B = configs.reshape([-1, self.nv]).astype(np.bool)
        
        term_1 = np.matmul(B, self.a)
        term_2 = utils.log1pexp(self.b + np.matmul(B, self.W)).sum(axis=1)

        if self.extra_hs:
            term_3 = utils.log1pexp(np.matmul(B, self.W_), keepdims=True).sum(axis=1)
        else:
            term_3 = 0
        
        logpsi = self.C + term_1 + term_2 + term_3
        
        return logpsi if B.shape[0] != 1 else logpsi.item()
    
    def mcmc_iter(self, init, n_steps, state=None, n=None, verbose=False):
        
        assert init.ndim==1 and len(init)==self.nv, "Invalid input."
        
        if state is None:
            logp = lambda x: 2*np.real(self(x))
            
        elif isinstance(state, str):
            assert isinstance(n, int), 'n has to be an int!'
            
            if state.lower()[0] == 'h':
                logp = lambda x: 2*np.real(self.eval_H(n, x))
            elif state.lower()[0] == 'x':
                logp = lambda x: 2*np.real(self.eval_X(n, x))
            elif state.lower()[0] == 'z':
                logp = lambda x: 2*np.real(self.eval_Z(n, x))
            else:
                raise NotImplementedError('State {} not recognized.'.format(state))
                
        else:
            raise TypeError('State has to be a string, {} given,'.format(type(state)))
        
        previous = init
        log_prob_old = logp(previous)
        
        accept_counter = 0
        
        for t in range(1, n_steps):
            
            i = np.random.randint(low=0, high=self.nv)
            
            proposal = previous.copy()
            proposal[i] = not proposal[i]
            
            log_prob_new = logp(proposal)
            
            logA = log_prob_new - log_prob_old
            
            if logA >= np.log(np.random.rand()):
                previous = proposal.copy()
                log_prob_old = log_prob_new
                accept_counter += 1
                
                yield proposal
            else:
                yield previous
        
        if verbose:
            print("Acceptance ratio: ", accept_counter/(n_steps-1))
            
    def get_samples(self, init, n_steps, state=None, n=None, verbose=False):
        return np.stack(list(self.mcmc_iter(init, n_steps, state, n, verbose)))
    
    def grad_log(self, configs):
        
        B = configs.reshape([-1, self.nv]).astype(np.bool)
        
        ga = configs.copy()
        gb = sigmoid(self.b + np.matmul(B, self.W))
        gW = np.matmul(ga[:, :, np.newaxis], gb[:, np.newaxis, :])
        
        return ga, gb, gW
        
    def hilbert_iter(self):
        for n in range(2**self.nv):
            yield np.fromiter(map(int, np.binary_repr(n, width=self.nv)), dtype=np.bool, count=self.nv)
            
    def get_state_vector(self, log=True, normalized=False):
        
        logpsis = np.fromiter(map(self, self.hilbert_iter()), dtype=np.complex, count=2**self.nv)
        
        if not normalized:
            
            if log:
                return logpsis
            else:
                return np.exp(logpsis, dtype=np.complex256)
        else:
            logZ = logsumexp(2*np.real(logpsis))
            
            if log:
                return logpsis - 0.5*logZ
            else:
                return np.exp(logpsis - 0.5*logZ, dtype=np.complex256)
        
    def get_lognorm(self, method='mcmc', samples=None, **mcmc_kwargs):

        if method.lower().startswith('ex'):
            logpsis = np.fromiter(map(self, self.hilbert_iter()), dtype=np.complex256, count=2**self.nv)
            return logsumexp(2*np.real(logpsis))

        elif method.lower().startswith('mc'):
            
            if samples is None:
                samples = self.get_samples(**mcmc_kwargs)

            logpsis = self(samples)
            return logsumexp(2*np.real(logpsis))

        else:
            raise NotImplementedError('Wrong method/ Expected "mcmc" or "exact", got {}'.format(method))
    
    def fidelity(self, other, method='mcmc', samples=None, other_samples=None, **mcmc_kwargs):

        if method.lower() == 'exact':
            bra = self.get_state_vector(log=False, normalized=True)
            ket = other.get_state_vector(log=False, normalized=True)

            return np.abs(np.vdot(bra, ket))**2

        elif method.lower() == 'mcmc':

            if samples is None:
                samples = self.get_samples(**mcmc_kwargs)

            if other_samples is None:
                other_samples = other.get_samples(**mcmc_kwargs)

            log_phi_over_psi = logsumexp(other(samples) - self(samples)) - np.log(samples.shape[0])
            log_psi_over_phi = logsumexp(self(other_samples) - other(other_samples)) - np.log(other_samples.shape[0])

            logF = log_phi_over_psi + log_psi_over_phi

            return np.real(np.exp(logF))

        else:
            raise NotImplementedError('Wrong method. Expected "mcmc" or "exact", got {}'.format(method))
    
    def eval_X(self, n, configs):
        
        aX = self.a.copy()
        aX[n] = -aX[n]
        bX = self.b + self.W[n,:]
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
        return logsumexp([self.eval_X(n, configs), self.eval_Z(n, configs)], axis=0) - np.log(2)/2

    def X(self, n):
        """
        Applies the Pauli X gate to qubit n.
        """
        self.a[n] = -self.a[n]
        self.b += self.W[n,:]
        self.W[n,:] = -self.W[n,:]
        self.C += self.a[n]

    def Y(self, n):
        """
        Applies the Pauli Y gate to qubit n.
        """
        self.a[n] = -self.a[n] + 1j*np.pi
        self.b += self.W[n,:]
        self.W[n,:] = -self.W[n,:]
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

    def _add_hidden_unit(self, k, l, Wkc, Wlc):

        ### NOT TESTED YET !!!

        if {k,l} not in self.extra_hs:
            W_ = np.zeros(shape=[self.nv, len(self.extra_hs)+1], dtype=np.complex)

            W_[:,:-1] = self.W_
            W_[k,-1] = Wkc
            W_[l,-1] = Wlc

            self.W_ = W_
            self.extra_hs.append({k,l})
        else:
            i = self.extra_hs.index({k,l})

            self.W_[k,i] += Wkc
            self.W_[l,i] += Wlc 
        
    def RZZ(self, k, l, phi):
        B = np.arccosh(np.exp(1j*phi))

        self._add_hidden_unit(k, l, -2*B, 2*B)
        self.a[k] += B
        self.a[l] -= B 

    def CRZ(self, k, l, phi):
        A = np.arccosh(np.exp(-1j*phi/2))

        self._add_hidden_unit(k, l, -2*A, 2*A)
        self.a[k] += 1j*phi/2 + A
        self.a[l] += 1j*phi/2 - A 