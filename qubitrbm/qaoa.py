import numpy as np
import networkx as nx

import os, sys
from time import time

import sympy, cirq
    
libpath = os.path.abspath('..')
if libpath not in sys.path:
    sys.path.append(libpath)

import qubitrbm.utils as utils

class QAOA:
    
    def __init__(self, graph, p=1):

        """
        A class that wraps around Cirq simulators to extract interesting information about smaller QAOA instances.

        Constructor arguments:

        graph: networkx.Graph
            The underlying QAOA graph

        p: Int
            The QAOA depth.
        """
        
        self.n_qubits = len(graph.nodes)
        self.qubits = cirq.LineQubit.range(self.n_qubits)

        assert p > 0, 'p has to be a positive integer!' 

        self.__p = int(p)
        self.graph = graph.copy()
        
        self.sim = cirq.Simulator(dtype=np.complex)

    def __reset_circuit(self):

        self.__gamma_params = []
        self.__beta_params = []

        self.__circuit = cirq.Circuit(cirq.H.on_each(self.__circuit_graph.nodes()))

        for p_ in range(self.__p):

            gamma = sympy.Symbol('gamma_{}'.format(p_))
            beta = sympy.Symbol('beta_{}'.format(p_))

            self.__gamma_params.append(gamma)
            self.__beta_params.append(beta)

            self.__circuit.append( (cirq.ZZPowGate(exponent=2*self.__gamma_params[-1]/np.pi)(u, v) for u, v in self.__circuit_graph.edges()) )
            self.__circuit.append(cirq.Moment(cirq.rx(rads=2*self.__beta_params[-1])(qubit) for qubit in self.__circuit_graph.nodes()))

        self.__circuit.append((cirq.measure(qubit) for qubit in self.__circuit_graph.nodes()))

    @property
    def circuit(self):
        """
        The underlying Cirq.Circuit object representing the specific QAOA circuit.
        """
        return self.__circuit

    @property
    def graph(self):
        """
        The underlying networkx.Graph object representing the specific QAOA graph.
        """
        return self.__graph

    @graph.setter
    def graph(self, G):

        assert isinstance(G, nx.Graph), 'Expected a nx.Graph instance, got {}'.format(type(graph))

        self.__graph = G.copy()
        self.__circuit_graph = nx.relabel_nodes( G, mapping={i: self.qubits[i] for i in range(self.n_qubits)} )

        if self.__p == 1:
            triangles = [c for c in nx.enumerate_all_cliques(G) if len(c)==3]
            self.__deltas = [len([t for t in triangles if u in t and v in t]) for u, v in G.edges()]

        self.__reset_circuit()

    @property
    def p(self):
        """
        The QAOA depth p.
        """
        return self.__p

    @p.setter
    def p(self, p_):
        assert p_ > 0, 'p has to be a positive integer!'
        self.__p = int(p_)
        self.__reset_circuit()

    def __get_param_resolver(self, gamma, beta):
        gamma_dict = dict(zip(self.__gamma_params, np.atleast_1d(gamma)))
        beta_dict = dict(zip(self.__beta_params, np.atleast_1d(beta)))
        return cirq.ParamResolver({**gamma_dict, **beta_dict})

    def sample(self, gamma, beta, n_samples):
        """
        Get samples from the underlying Cirq quantum cirquit simulator.

        gamma: float (if self.p==1) or numpy.array of shape [self.p]
            Values of QAOA \gamma parameters.

        beta: float (if self.p==1) or numpy.array of shape [self.p]
            Values of QAOA \beta parameters.

        n_samples: Int
            The number of samples required.

        Returns: A numpy.array of bitstrings with shape [n_samples, n_qubits].
        """
        param_res = self.__get_param_resolver(gamma, beta)
        res = self.sim.sample(program=self.__circuit, params=param_res, repetitions=n_samples)
        return res.drop(labels=[col for col in res.columns if 'gamma' in col or 'beta' in col], axis=1).values

    def simulate(self, gamma, beta):
        """
        Get the full normalized output state of the QAOA circuit.

        gamma: float (if self.p==1) or numpy.array of shape [self.p]
            Values of QAOA \gamma parameters.

        beta: float (if self.p==1) or numpy.array of shape [self.p]
            Values of QAOA \beta parameters.

        Returns: A cirq.StateVectorTrialResult object.
        (Full array of amplitudes can be accessed with the .final_state_vector method.)
        """
        param_res = self.__get_param_resolver(gamma, beta)
        return self.sim.simulate(self.__circuit[:-1], param_resolver=param_res, qubit_order=self.qubits)

    def cost(self, configs):
        """
        Calculate the MaxCut cost for given bitstrings

        configs: numpy.array of shape [batch_size, qubit_number]
            The batch of bitstrings to calculate MaxCut costs for.

        Returns: A 1D numpy.array of cost values corresponding to rows in `configs`.
        """
        B = np.atleast_2d(configs)
        z = (-1)**B[:, self.graph.edges()]
        return z.prod(axis=-1).sum(axis=-1)

    def exact_cost(self, gamma, beta):

        """
        Calculate the exact value of the MaxCut cost at given angles if p=1.

        gamma: float or numpy.array of shape [batch_size]
            Values of QAOA \gamma parameters.

        beta: float or numpy.array of shape [batch_size]
            Values of QAOA \beta parameters.

        Returns: A float or 1D numpy.array of cost values corresponding to matching entries in `gamma` and `beta`.
        """
        
        assert self.p == 1, 'The exact formula holds only for p=1!'

        C = 0.0
        s2g, c2g, c4g = np.sin(2*gamma), np.cos(2*gamma), np.cos(4*gamma)
        s4b, s2b = np.sin(4*beta), np.sin(2*beta)
        
        for (u, v), D in zip(self.graph.edges(), self.__deltas):

            du = self.graph.degree(u) - 1
            dv = self.graph.degree(v) - 1
            
            term1 = s4b*s2g*(c2g**du + c2g**dv)
            term2 = (s2b**2)*(c2g**(du+dv-2*D))*(1 - c4g**D)

            C += (term1 + term2)/2
        
        return C

    def cost_from_probs(self, probs, hilbert=None):

        """
        Calculates cost from all bitstring probabilities. A helper function.

        probs: numpy.array
            An array of all 2**n_qubits probability values that should sum to 1.

        hilbert: 2D numpy.array or None (default=None)
            A 2D array containing all 2**self.nv bitstrings for the given graph.
            (If not provided, it will be calculated using qubitrbm.utils.hilbert_iter.)

        """

        if hilbert is None:
            hilbert  = np.array(list(utils.hilbert_iter(self.n_qubits))) 

        return np.sum(probs*self.cost(hilbert))

    def num_cost_from_params(self, gamma, beta, method='exact', **kwargs):

        """
        Calculate the MaxCut cost using either the exact output wavefunction from the underlying Cirq simulator or estimate the cost using output state samples.
        
        gamma: float (if self.p==1) or numpy.array of shape [self.p]
            Values of QAOA \gamma parameters.

        beta: float (if self.p==1) or numpy.array of shape [self.p]
            Values of QAOA \beta parameters.

        method: str
            Either 'exact' or 'sample', choosing between the two methods of calculating the cost.

        hilbert: 2D numpy.array or None (default=None) [Used only if method='exact']
            A 2D array containing all 2**self.nv bitstrings for the given graph.
            (If not provided, it will be calculated using qubitrbm.utils.hilbert_iter.)

        n_samples: Int [Used only of method='sample']
            The number of samples required.
        """

        if method.lower() == 'exact':
            psi = self.simulate(gamma, beta).final_state
            p = np.abs(psi)**2
            return self.cost_from_probs(p, **kwargs)

        elif method.lower().startswith('sampl'):
            samples = self.sample(gamma, beta, **kwargs)
            return self.cost(samples)

        else:
            raise KeyError('Expected "exact" or "sampling" for method, got {}'.format(method))

    def num_grad_cost(self, gamma, beta, dx=1e-5, hilbert=None):
        """
        Calculate the finite-difference estimator of the derivative of the MaxCut cost with respect to the QAOA angles.

        gamma: float (if self.p==1) or numpy.array of shape [self.p]
            Values of QAOA \gamma parameters.

        beta: float (if self.p==1) or numpy.array of shape [self.p]
            Values of QAOA \beta parameters.

        dx: float
            The finite difference for angles.

        Returns:
            A 1d numpy.array of length 2p: first p elements represent gradients w.r.t. QAOA \gamma parameters and the second p elements represent gradients w.r.t. QAOA \beta parameters.
        """

        gs = np.atleast_1d(gamma)
        bs = np.atleast_1d(beta)

        grad = np.empty(shape=2*self.p)

        if hilbert is None:
            hilbert  = np.array(list(utils.hilbert_iter(self.n_qubits)))

        for j, (g, b) in enumerate(zip(gs, bs)):
            
            e = np.zeros(self.p)
            e[j] = 1

            gamma_p = self.num_cost_from_params(gamma + e*dx/2, beta, hilbert=hilbert)
            gamma_m = self.num_cost_from_params(gamma - e*dx/2, beta, hilbert=hilbert)
            grad[j] = (gamma_p - gamma_m)/dx

            beta_p = self.num_cost_from_params(gamma, beta + e*dx/2, hilbert=hilbert)
            beta_m = self.num_cost_from_params(gamma, beta - e*dx/2, hilbert=hilbert)
            grad[self.p + j] = (beta_p - beta_m)/dx

        return grad

    def exact_grad_cost(self, gamma, beta):

        """
        Efficiently calculate the finite-difference estimator of the derivative of the MaxCut cost with respect to the QAOA angles if p=1.

        gamma: float or numpy.array of shape [batch_size]
            Values of QAOA \gamma parameters.

        beta: float or numpy.array of shape [batch_size]
            Values of QAOA \beta parameters.

        Returns:
            A numpy.array of shape [batch_size,2] containing gradients w.r.t. gamma & beta.
        
        """

        assert self.p == 1, 'The exact formula holds only for p=1!'

        d_beta, d_gamma = 0.0, 0.0

        c2g, c4g, s2g, s4g = np.cos(2*gamma), np.cos(4*gamma), np.sin(2*gamma), np.sin(4*gamma)
        s2b, s4b, c4b = np.sin(2*beta), np.sin(4*beta), np.cos(4*beta)

        for (k, l), D in zip(self.graph.edges(), self.__deltas):
            
            dk = self.graph.degree(k) - 1
            dl = self.graph.degree(l) - 1

            gamma_term_1 = s4b * c2g * (c2g**dk + c2g**dl)
            gamma_term_2 = -(dk+dl-2*D) * s2b**2 * s2g * c2g**(dk+dl-2*D-1) * (1-c4g**D)
            gamma_term_3 = 2*D * (s2b**2) * s4g * c2g**(dk+dl-2*D) * c4g**(D-1)
            gamma_term_4 = -s4b * s2g**2 * (dk* c2g**(dk-1) + dl* c2g**(dl-1))

            d_gamma += gamma_term_1 + gamma_term_2 + gamma_term_3 + gamma_term_4

            beta_term_1 = 2 * c4b * s2g * (c2g**dk + c2g**dl)
            beta_term_2 = s4b * c2g**(dk+dl-2*D) * (1-c4g**D)

            d_beta +=  beta_term_1 + beta_term_2

        return np.array([d_gamma, d_beta], dtype=np.float)

    def optimize(self, lr=1e-3, tol=1e-3, init=None, betas=(0.9, 0.999), eps=1e-6, hilbert=None, dx=1e-5, verbose=True):

        """
        Use the Adam optimizer to find the minimum of the MaxCut cost as a function of QAOA parameters. If QAOA.p is set to 1, then fast exact cost values and fast exact gradients will be used automatically. Otherwise, costs are evaluated through circuit simulation and gradients through finite difference approximatiomns.

        lr: float
            The Adam learning rate parameter.

        tol: float
            Sets the tolerance for stopping the optimization - when two consecutive cost values are less than `tol` apart, the optimization loop exits.
        
        init: numpy.array or None
            Initial angles to start the optimization from. The first half is interpreted as gammas and the second as betas.

        betas: tuple of 2 floats
            Beta parameters for Adam.

        eps: float
            The \epsilon parameter for Adam.

        hilbert: 2D numpy.array or None (default=None) [Used only if p>1]
            A 2D array containing all 2**self.nv bitstrings for the given graph.
            (If not provided, it will be calculated using qubitrbm.utils.hilbert_iter.)

        dx: float [Used only if p>1]
            The finite difference for gradient estimators.

        verbose: bool
            Whether to print out optimizer progress report every 10 seconds.

        Returns: A 2-tuple of numpy.arrays
            1) Optimal angles of shape init.shape.
            2) MaxCut costs as a function of optimization step (1D array of shape [# of optimizer steps taken])
        """

        beta1, beta2 = betas
        
        if init is None:
            gamma = np.random.uniform(0, np.pi/2, size=self.p)
            beta = np.random.uniform(-np.pi/4, np.pi/4, size=self.p)
            params = np.concatenate([gamma, beta])
        else:
            params = np.asarray_chkfinite(init, dtype=float)

        if hilbert is None and self.p > 1:
            hilbert = np.array(list(utils.hilbert_iter(self.n_qubits)))

        g, b = np.split(params, 2) if self.p > 1 else params

        m = np.zeros_like(params)
        v = np.zeros_like(params)
        t = 0
        clock = time()
        history = []

        while True:

            t += 1

            if self.p > 1:
                grad = self.num_grad_cost(g, b, hilbert=hilbert, dx=dx)
            else:
                grad = self.exact_grad_cost(g, b)

            m = beta1*m + (1-beta1)*grad
            v = beta2*v + (1-beta2)*grad**2

            m_hat = m/(1 + beta1**t)
            v_hat = v/(1 + beta2**t)

            d_params = lr * m_hat/(np.sqrt(v_hat) + eps)

            if np.any(d_params > tol):
                params -= d_params
            else:
                break

            g, b = np.split(params, 2) if self.p > 1 else params

            if self.p > 1:
                f = self.num_cost_from_params(g, b, hilbert=hilbert)
            else:
                f = self.exact_cost(g, b)

            history.append(f)

            if time() - clock > 10 and verbose:
                print('Iteration {:3d} | Cost = {:05.4f}'.format(t, f))
                clock = time()

        return params, np.array(history)