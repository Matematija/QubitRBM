import numpy as np
import networkx as nx

import cirq
import sympy

import pybobyqa as optim

import os, sys

libpath = os.path.abspath('..')
if libpath not in sys.path:
    sys.path.append(libpath)

import QubitRBM.utils as utils

class QAOA:
    
    def __init__(self, graph, p=1):
        
        self.n_qubits = len(graph.nodes)
        self.p = p
        
        self.qubits = cirq.LineQubit.range(self.n_qubits)
        self.graph = graph.copy()
        self.circuit_graph = nx.relabel_nodes(graph, mapping={i: self.qubits[i] for i in range(self.n_qubits)})
        
        self.gamma_params = []
        self.beta_params = []

        self.circuit = cirq.Circuit(cirq.H.on_each(self.circuit_graph.nodes()))

        for p_ in range(self.p):

            gamma = sympy.Symbol('gamma_{}'.format(p_))
            beta = sympy.Symbol('beta_{}'.format(p_))

            self.gamma_params.append(gamma)
            self.beta_params.append(beta)

            self.circuit.append( (cirq.ZZPowGate(exponent=2*self.gamma_params[-1]/np.pi)(u, v) for u, v in self.circuit_graph.edges()) )
            self.circuit.append(cirq.Moment(cirq.rx(rads=2*self.beta_params[-1])(qubit) for qubit in self.circuit_graph.nodes()))

        self.circuit.append((cirq.measure(qubit) for qubit in self.circuit_graph.nodes()))
        
        self.sim = cirq.Simulator(dtype=np.complex128)

    def _get_param_resolver(self, gamma, beta):
        gamma_dict = dict(zip(self.gamma_params, np.atleast_1d(gamma)))
        beta_dict = dict(zip(self.beta_params, np.atleast_1d(beta)))
        return cirq.ParamResolver({**gamma_dict, **beta_dict})

    def _get_param_bounds_for_optim(self, gamma_lims=(0, np.pi/2), beta_lims=(-np.pi/4, np.pi/4)):

        gl, gu = gamma_lims
        bl, bu = beta_lims

        lower = np.concatenate([gl*np.ones(self.p), bl*np.ones(self.p)])
        upper = np.concatenate([gu*np.ones(self.p), bu*np.ones(self.p)]) 

        return (lower, upper)

    def sample(self, gamma, beta, n_samples):
        param_res = self._get_param_resolver(gamma, beta)
        res = self.sim.sample(program=self.circuit, params=param_res, repetitions=n_samples)
        return res.drop(labels=[col for col in res.columns if 'gamma' in col or 'beta' in col], axis=1).values

    def simulate(self, gamma, beta):
        param_res = self._get_param_resolver(gamma, beta)
        return self.sim.simulate(self.circuit[:-1], param_resolver=param_res, qubit_order=self.qubits)

    def cost(self, configs):
        B = np.atleast_2d(configs)
        return ((-1)**B[:, self.graph.edges()]).prod(axis=-1).sum(axis=-1)

    def cost_from_params(self, gamma, beta, method='exact', **kwargs):

        if method.lower() == 'exact':
            psi = self.simulate(gamma, beta).final_state
            p = np.abs(psi)**2
            return self.cost_from_probs(p, **kwargs)

        elif method.lower().startswith('sampl'):
            samples = self.sample(gamma, beta, **kwargs)
            return self.cost(samples)

        else:
            raise KeyError('Expected "exact" or "sampling" for method, got {}'.format(method))

    def cost_from_probs(self, probs, hilbert=None):

        if hilbert is None:
            hilbert  = np.array(list(utils.hilbert_iter(self.n_qubits))) 

        return np.sum(probs*self.cost(hilbert))
    
    def optimize(self, init=None, method='exact', **kwargs):

        if init is None:
            gamma = np.random.uniform(0, np.pi/2, size=self.p)
            beta = np.random.uniform(-np.pi/4, np.pi/4, size=self.p)
            init = np.concatenate([gamma, beta])

        objfun_kwargs = {key: val for key, val in kwargs.items() if key in ['hilbert', 'n_samples']}
        optim_kwargs = {key: val for key, val in kwargs.items() if key not in ['hilbert', 'n_samples']}

        has_noise = False if method.lower() == 'exact' else True
        objfun = lambda params: self.cost_from_params(params[:self.p], params[self.p:], method=method, **objfun_kwargs)

        bounds = self._get_param_bounds_for_optim()

        return optim.solve(objfun=objfun, x0=init, bounds=bounds, objfun_has_noise=has_noise, **optim_kwargs)

    def grad_cost(self, gamma, beta, dx=1e-4, hilbert=None):

        gs = np.atleast_1d(gamma)
        bs = np.atleast_1d(beta)

        grad = np.empty(shape=2*self.p)

        if hilbert is None:
            hilbert  = np.array(list(utils.hilbert_iter(self.n_qubits)))

        for j, (g, b) in enumerate(zip(gs, bs)):
            
            e = np.zeros(self.p)
            e[j] = 1

            gamma_p = self.cost_from_params(gamma + e*dx/2, beta, hilbert=hilbert)
            gamma_m = self.cost_from_params(gamma - e*dx/2, beta, hilbert=hilbert)
            grad[j] = (gamma_p - gamma_m)/dx

            beta_p = self.cost_from_params(gamma, beta + e*dx/2, hilbert=hilbert)
            beta_m = self.cost_from_params(gamma, beta - e*dx/2, hilbert=hilbert)
            grad[self.p + j] = (beta_p - beta_m)/dx

        return grad