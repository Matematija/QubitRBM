import numpy as np
from scipy.special import logsumexp
from collections import OrderedDict

from numba import njit
import os, sys

libpath = os.path.abspath('..')
if libpath not in sys.path:
    sys.path.append(libpath)

import qubitrbm.utils as utils
from qubitrbm.utils import log1pexp, logaddexp

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
        """
        A Restricted Boltzmann Machine class.

        n_visible: Int
            Number of visible units.

        n_hidden: Int
            Number of hidden units.
        """
        
        self.__nv = int(n_visible)
        self.__nh = int(n_hidden)
        
        self.W = np.zeros([self.nv, self.nh], dtype=complex)
        self.a = np.zeros(self.nv, dtype=complex)
        self.b = np.zeros(self.nh, dtype=complex)
        self.C = -(self.nh + self.nv/2)*np.log(2)

        self.mask = np.ones(shape=self.W.shape, dtype=np.bool)

        self.__eval_from_params = _eval_RBM_from_params
        self.__eval_RX_from_params = _eval_RX_RBM_from_params

    def __repr__(self):
        lines = []
        lines.append('Qubit RBM with:')
        lines.append('\t{} visible units, {} hidden units'.format(self.nv, self.nh))
        lines.append('\t{} free parameters out of {} in total'.format(self.n_free_par, self.n_par))

        return '\n'.join(lines)

    def __call__(self, configs, fold=True, squeeze=True):

        """
        Evaluates the natural logarithm of the unnormalized wavefunction.

        configs: numpy.array of shape [batch_size, self.nv]
            Batched input of classical bitstrings.

        Returns: A 1D numpy.array of values representing logarithms of the unnormalized RBM wavefunction.
        """
        
        B = np.atleast_2d(configs).astype(complex)
        logpsi = self.__eval_from_params(B, self.C, self.a, self.b, self.W)

        if fold:
            logpsi.imag = (logpsi.imag + np.pi)%(2*np.pi) - np.pi
        
        # return logpsi if logpsi.size > 1 else logpsi.item() if squeeze else logpsi
        return logpsi

    @property
    def nv(self):
        """
        The number of visible units in the RBM.
        """
        return self.__nv
    
    @property
    def nh(self):
        """
        The number of hidden units in the RBM.
        """
        return self.__nh

    @property
    def C(self):
        """
        The log of the numerical prefactor for wavefunction evaluation.
        Changing this shouldn't affect any physical observables, it's just here for optional tweaking for numerical purposes.
        """
        return self.__C

    @C.setter
    def C(self, C):
        self.__C = complex(C)

    @property
    def a(self):
        """
        The visible biases of the RBM.

        Returns: A 2D numpy.array of shape [self.nv].
        """
        return self.__a
    
    @a.setter
    def a(self, a):
        self.__a = np.asarray_chkfinite(a, dtype=complex).reshape(self.nv)
    
    @property
    def b(self):
        """
        The hidden biases of the RBM.

        Returns: A 1D numpy.array of shape [self.nh].
        """
        return self.__b
    
    @b.setter
    def b(self, b):
        self.__b = np.asarray_chkfinite(b, dtype=complex).reshape(self.nh)

    @property
    def W(self):
        """
        The weight matrix of the RBM.

        Returns: A 2D numpy.array of shape [self.nv, self.nh].
        """
        return self.__W

    @W.setter
    def W(self, W):
        self.__W = np.asarray_chkfinite(W, dtype=complex).reshape(self.nv, self.nh)

    @property
    def mask(self):
        """
        A boolean matrix of the same shape as self.W indicating which parameters are being optimized over. Setting mask[i,j]=False "deactivates" W[i,j] with the idea of providing an alternative way (to compression) of handling the hidden unit count increase.
        """
        return self.__mask

    @mask.setter
    def mask(self, mask):
        self.__mask = np.asarray_chkfinite(mask, dtype=bool).reshape(self.nv, self.nh)

    @property
    def params(self):
        """
        A convenience property containing all parameters. RBM.params = [a, b, W.reshape(-1)]
        """
        return np.concatenate([self.a, self.b, self.W[self.mask]], axis=0)

    @params.setter
    def params(self, params):
        assert len(params) == self.n_free_par, 'Invalid number of parameters given. {} expected, got {}.'.format(self.n_free_par, len(params))
        self.a = params[:self.nv].copy()
        self.b = params[self.nv:(self.nv + self.nh)].copy()
        self.W[self.mask] = params[(self.nv + self.nh):].copy()

    @property
    def state_dict(self):
        """
        A dictionary containing all parameters.

        Returns: An OrderedDict with
            keys: ['C', 'a', 'b', 'W']
            values: numpy.arrays of corresponding parameters.

        Note: The corresponding setter (RBM.state_dict = d) is the only way to change the number of hidden units on-the-fly, without constructing a new RBM object.
        """
        return OrderedDict(zip(['C', 'a', 'b', 'W'], [self.C, self.a, self.b, self.W]))

    @state_dict.setter
    def state_dict(self, sd):

        a = np.asarray_chkfinite(sd['a'], dtype=complex).reshape(self.nv)
        b = np.asarray_chkfinite(sd['b'], dtype=complex).reshape(-1)
        W = np.asarray_chkfinite(sd['W'], dtype=complex).reshape(self.nv, -1)

        assert len(b) == W.shape[1], 'Inconsistent number of hidden units.'

        self.__a, self.__b, self.__W = a, b, W
        self.__nv, self.__nh = W.shape

        if 'mask' in sd.keys():
            self.mask = sd['mask']
        else:
            self.__mask = np.ones_like(W, dtype=bool)

        if 'C' in sd.keys():
            self.C = sd['C']
        else:
            self.C = -(self.nh + self.nv/2)*np.log(2)

    @property
    def n_free_par(self):
        """
        The number of free parameters in the RBM, `excluding` all those deactivated by setting elements of self.mask to False.

        Returns: Int
        """
        return self.nv + self.nh + self.mask.sum()

    @property
    def n_par(self):
        """
        The number of free parameters in the RBM, `including` all those deactivated by setting elements of self.mask to False.

        Returns: Int
        """
        return self.nv + self.nh + self.nv*self.nh

    @property
    def alpha(self):
        """
        The hidden unit density: n_hidden/n_visible.
        """
        return self.nh/self.nv
            
    def fold_imag_params(self):
        """
        Translates (in-place) imaginary parts of all parameter values to the [-pi, pi] range.

        Returns: None
        """
        self.C = utils.fold_imag(self.C)
        self.params = utils.fold_imag(self.params)

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
        """
        Samples the RBM with the single-spin flip MH algorithm.

        n_steps: Int
            The number of samples to obtain.
        
        state: str or None (default=None)
            Either None or "rx". None samples the RBM itself and "rx" samples exp(-i*\beta X_n) |RBM>
        
        init: numpy.array of shape [n_chains,self.nv] (default=None)
            Initial bitstrings to start sampling from in each independent chain. Defaults to a set of random bitstrings.
        
        n_chains: Int (default=1)
            The number of independent Markov chains to run.
            
        warmup: Int (default=0)
            The number of consecutive samples to discard in each Markov chain to facilitate equilibriation,
            
        step: Int (default=1)
            The number of MCMC steps to take in between samples. (step=1 means record each sample, step=2 means record every other...)
            
        T: float (default=1.0)
            The temperature to use when sampling, possibly with algorithms such as parallel tempering. Is not internally used and was implemented as an experiment.
        
        verbose: bool (default=False)
            Whether to print out sampling information such as the current step and acceptance rate.
        
        n: Int [Only used if state='rx']
            Which qubit to apply the RX gate to and then sample. In other words, it's the n in exp(-i*\beta X_n) |RBM> when we wish to sample that state.

        beta: float [Only used if state='rx']
            The angle of the RX gate. In other words, it's the \beta in exp(-i*\beta X_n) |RBM> when we wish to sample that state.
        
        Returns: A 2D numpy.array of shape [n_chains*n_steps, self.nv] with samples (bitstrings) from all chains.
        """

        samples = np.array(list(self.iter_samples(*args, **kwargs)))
        return samples.reshape(-1, self.nv)

    def get_exact_samples(self, n_samples, state=None, hilbert=None, **kwargs):

        """
        Samples the state exactly using numpy.random.choice. Recommended only for small qubit counts.

        n_samples: Int
            The number of samples to obtain.

        state: str or None (default=None)
            Either None or "rx". None samples the RBM itself and "rx" samples exp(-i*\beta X_n) |RBM>

        hilbert: 2D numpy.array or None (default=None)
            A 2D array containing all 2**self.nv bitstrings for the given graph.
            (If not provided, it will be calculated using qubitrbm.utils.hilbert_iter.)

        kwargs:
            Additional keyword arguments to forward to RBM.eval_RX if state='rx'.

        Returns: A 2D numpy.array of shape [n_samples, self.nv] containing exactly sampled bitstrings.
        """

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
        """
        Calculates the gradient of the log-wavefunction

        configs: 1D or 2D numpy.array of shape [self .nv] or [batch_size, self.nv]
            Input bitstrings for gradient calculation.

        Returns: A 1D numpy.array of shape [self.n_free_par] if configs is 1D or a 2D numpy.array of shape [batch_size, self.n_free_par] if configs is 2D.
        """

        B = np.atleast_2d(configs).astype(complex)
        ga, gb, gW = self.__grad_log_from_params(B, self.a, self.b, self.W)
        return np.concatenate([ga, gb, gW[:,self.mask]], axis=1)
            
    def get_state_vector(self, normalized=False, state=None, hilbert=None, **kwargs):
        """
        Calculates the full state vector of the RBM wavefunction.

        normalized: bool
            Determines whether to normalize the output wavefunction. If false, a vector of complex log-values of the wavefunction is returned.

        state: str or None (default=None)
            Either None or "rx". None samples the RBM itself and "rx" samples exp(-i*\beta X_n) |RBM>

        hilbert: 2D numpy.array or None (default=None)
            A 2D array containing all 2**self.nv bitstrings for the given graph.
            (If not provided, it will be calculated using qubitrbm.utils.hilbert_iter.)

        Returns: A 1D numpy.array of shape [2**self.nv] containig all bitstring amplitudes corresponding to rows of `hilbert`.
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
    
    def eval_X(self, configs, n, fold=True):
        """
        Evaluates log-values of the current RBM state after applying the Pauli X gate on qubit n. Does not change parameters in-place.

        configs: 1D or 2D numpy.array of shape [self .nv] or [batch_size, self.nv]
            Input bitstrings for amplitude evaluation.

        n: Int
            Determines the qubit to apply the X gate on.

        fold: Bool
            Whether to "fold" the imaginary values of output log-amplitudes to be between [-pi, pi].
            (Introduced for debugging purposes, should not make any difference.)

        Returns: A 1D numpy.array of shape [batch_size] containing log-amplitudes corresponding to bitstrings in `configs`.
        """
        
        CX, aX, bX, WX = self.__get_X_params(n)

        B = np.atleast_2d(configs).astype(complex)
        res = self.__eval_from_params(B, CX, aX, bX, WX)

        if fold:
            return utils.fold_imag(res)
        else:
            return res
        
    def eval_Z(self, configs, n, fold=True):
        """
        Evaluates log-values of the current RBM state after applying the Pauli Z gate on qubit n. Does not change parameters in-place.

        configs: 1D or 2D numpy.array of shape [self .nv] or [batch_size, self.nv]
            Input bitstrings for amplitude evaluation.

        n: Int
            Determines the qubit to apply the X gate on.

        fold: Bool
            Whether to "fold" the imaginary values of output log-amplitudes to be between [-pi, pi].
            (Introduced for debugging purposes, should not make any difference.)

        Returns: A 1D numpy.array of shape [batch_size] containing log-amplitudes corresponding to bitstrings in configs.
        """
        
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
    
    # def eval_H(self, configs, n, fold=True):

    #     Xvals = self.eval_X(configs, n=n, fold=False)
    #     Zvals = self.eval_Z(configs, n=n, fold=False)

    #     b = (1/np.sqrt(2), 1/np.sqrt(2))
    #     res = logaddexp(Xvals, Zvals, b=b)

    #     if fold:
    #         return utils.fold_imag(res)
    #     else:
    #         return res

    def eval_RX(self, configs, n, beta, fold=True):
        """
        Evaluates log-values of the current RBM state after applying the RX gate (= exp(-i*\beta X_n) ) on qubit n. Does not change parameters in-place.

        configs: 1D or 2D numpy.array of shape [self .nv] or [batch_size, self.nv]
            Input bitstrings for amplitude evaluation.

        n: Int
            Determines the qubit to apply the RX gate on.

        beta: float
            The angle of the RX gate. In other words, it's the \beta in exp(-i*\beta X_n) |RBM>.

        fold: Bool
            Whether to "fold" the imaginary values of output log-amplitudes to be between [-pi, pi].
            (Introduced for debugging purposes, should not make any difference.)

        Returns: A 1D numpy.array of shape [batch_size] containing log-amplitudes corresponding to bitstrings in `configs`.
        """

        B = np.atleast_2d(configs).astype(complex)

        C, a, b, W = self.C, self.a, self.b, self.W
        CX, aX, bX, WX = self.__get_X_params(n)

        res = self.__eval_RX_from_params(B, C, a, b, W, CX, aX, bX, WX, beta)

        if fold:
            res.imag = (res.imag + np.pi)%(2*np.pi) - np.pi

        return res

    def X(self, n):
        """
        Applies the Pauli X gate to qubit n in-place.
        """
        self.a[n] = -self.a[n].copy()
        self.b += self.W[n,:].copy()
        self.W[n,:] = -self.W[n,:].copy()
        self.C -= self.a[n].copy()

    def Y(self, n):
        """
        Applies the Pauli Y gate to qubit n in-place.
        """
        self.a[n] = -self.a[n].copy() + 1j*np.pi
        self.b += self.W[n,:].copy()
        self.W[n,:] = -self.W[n,:].copy()
        self.C += self.a[n].copy() + 1j*np.pi/2

    def Z(self, n):
        """
        Applies the Pauli Z gate to qubit n in-place.
        """
        self.a[n] += 1j*np.pi

    def RZ(self, n, phi):
        """
        Applies the following unitary gate to qubit n: e^{-i Z_n \phi /2} ~ [[1, 0], [0, e^{i \phi}]] (in-place).
        """
        self.a[n] += 1j*phi

    def add_hidden_units(self, num, b_=None, W_=None, mask=False):
        """
        Adds hidden units to the RBM by extending hidden biases b and the weight matrix W.

        num: Int
            The number of hidden units to add.
        
        b_: numpy.array of shape [num] or None (default=None)
            A vector of additional hidden biases to stick to the tail end of self.b. If `None`, it's taken to be all zeros.

        W_: numpy.array of shape [self.nv, num] or None (default=None)
            A vector of additional weight matrix elements to stick to the tail end of self.W. If `None`, it's taken to be all zeros.

        mask: bool
            Wheteher to set newly introduced elements of self.mask to true or false.
            (Argument mask=True will mask these elements by setting the corresponding elements of self.mask to False.)
        """
        
        mask_old = self.mask

        if b_ is None: 
            b_ = np.zeros(shape=[num], dtype=complex) 
        if W_ is None:
            W_ = np.zeros(shape=[self.nv, num], dtype=complex)

        b = np.zeros(shape=[self.nh+num], dtype=complex)
        W = np.zeros(shape=[self.nv, self.nh+num], dtype=complex)

        b[:-num] = self.b
        b[-num:] = b_
        self.__b = b

        W[:,:-num] = self.W
        W[:,-num:] = W_
        self.__W = W
        self.C -= np.log(2)

        self.__nh += num

        if mask:
            m = np.zeros(shape=[self.nv, num], dtype=np.bool)
        else:
            m = np.ones(shape=[self.nv, num], dtype=np.bool)

        self.__mask = np.concatenate([self.__mask, m], axis=1)
        
    def RZZ(self, k, l, phi, mask=False):
        """
        Applies the two-qubit rotation operator exp(-i phi Z_k Z_l /2) in-place by introducing new hidden units.

        k, l: Int
            The two qubits to apply the rotation to.
        
        phi: float
            The angle of rotation.

        mask: bool
            Whether to mask (deactivate) the newly introduced weight matrix (self.W) elements.
        """

        self.add_hidden_units(num=1, mask=mask)
        self.mask[[k,l], -1] = True

        B = np.arccosh(np.exp(1j*phi))

        self.W[k,-1] = -2*B
        self.W[l,-1] = 2*B
        self.a[k] += B
        self.a[l] -= B
        # self.C += np.log(2) - 1j*phi/2
        self.C -= np.log(2)

    def UC(self, graph, gamma, mask=False):
        """
        Applies the QAOA U_C operator for each edge in a given graph.

        graph: networkx.Graph
            The underlying QAOA graph to take edges from by iterating over graph.edges(). Vertex variables should be Ints 

        gamma: float
            The QAOA angle/parameter to use with the gate.

        mask: bool
            Whether to mask (deactivate) the newly introduced weight matrix (self.W) elements.
        """

        for u, v in graph.edges():
            self.RZZ(u, v, 2*gamma, mask=mask)

    # def CRZ(self, k, l, phi, mask=False):

    #     self.add_hidden_units(num=1, mask=mask)
    #     self.mask[[k,l], -1] = True

    #     A = np.arccosh(np.exp(-1j*phi/2))

    #     self.W[k,-1] = -2*A
    #     self.W[l,-1] = 2*A
    #     self.a[k] += 1j*phi/2 + A
    #     self.a[l] += 1j*phi/2 - A
    #     self.C -= np.log(2)

    def save(self, path, **kwargs):
        """
        Saves all of the model parameters in a numpy .npz file, along with optional additional arrays. This is a literal one-liner function:

        np.savez(path, **self.state_dict, mask=self.mask, **kwargs)

        path: str
            The path on the system to save the file to

        kwargs:
            Any additional arrays to store with self.state_dict and self.mask.
        """


        np.savez(path, **self.state_dict, mask=self.mask, **kwargs)

    def load(self, path):
        """
        Loads the RBM state from a numpy .npz file. After loading the archive, it looks for keys ["C", "a", "b", "W"] and stores corresponding values into internal variables.

        Returns: The loaded .npz file as an OrderedDict
        """
        sd = OrderedDict(np.load(path))
        self.state_dict = sd
        return sd