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
        self.graph = nx.relabel_nodes(graph, mapping={i: self.qubits[i] for i in range(self.n_qubits)})
        
        self.gamma_params = []
        self.beta_params = []
        self.param_bounds = np.array([[0, -np.pi/4], [np.pi/2, np.pi/4]])

        self.circuit = cirq.Circuit(cirq.H.on_each(self.graph.nodes()))

        for p_ in range(self.p):
            self.gamma_params.append(sympy.Symbol('gamma_{}'.format(p_)))
            self.beta_params.append(sympy.Symbol('beta_{}'.format(p_)))

            self.circuit.append( (cirq.ZZPowGate(exponent=2*self.gamma_params[-1]/np.pi)(u, v) for u, v in self.graph.edges()) )
            self.circuit.append(cirq.Moment(cirq.rx(rads=2*self.beta_params[-1])(qubit) for qubit in self.graph.nodes()))

        self.circuit.append((cirq.measure(qubit) for qubit in self.graph.nodes()))
        
        self.sim = cirq.Simulator(dtype=np.complex128)

    def _get_param_resolver(self, gamma, beta):
        gamma_dict = dict(zip(self.gamma_params, np.atleast_1d(gamma)))
        beta_dict = dict(zip(self.beta_params, np.atleast_1d(beta)))
        return cirq.ParamResolver({**gamma_dict, **beta_dict})

    def sample(self, gamma, beta, n_samples):
        param_res = self._get_param_resolver(gamma, beta)
        res = self.sim.sample(program=self.circuit, params=param_res, repetitions=n_samples)
        return res.drop(labels=['gamma', 'beta'], axis=1).values

    def simulate(self, gamma, beta):
        param_res = self._get_param_resolver(gamma, beta)
        return self.sim.simulate(self.circuit[:-1], param_resolver=param_res, qubit_order=self.qubits)

    def cost(self, samples):

        cost_value = 0.0

        for qi, qj in self.graph.edges():
            term = (-1)**(samples[:,qi.x]*samples[:,qj.x])
            cost_value += np.mean(term)

        return cost_value

    def cost_from_params(self, params, n_samples):
        samples = self.sample(*params, n_samples)
        return self.cost(samples)

    def cost_from_probs(self, p, hilbert=None):

        if hilbert is None:
            hilbert = np.array(list(utils.hilbert_iter(self.n_qubits)), dtype=np.int)

        cost = 0.0

        for qi, qj in self.graph.edges():
            term = p*(-1)**(hilbert[:,qi.x]*hilbert[:,qj.x])
            cost += term.sum()

        return cost
    
    def optimize(self, n_samples, init=None, **kwargs):

        if init is None:
            gamma = np.random.uniform(0, np.pi/2)
            beta = np.random.uniform(-np.pi/4, np.pi/4)
            init = np.array([gamma, beta])

        objfun = lambda params: self.cost_from_params(params, n_samples)

        return optim.solve(objfun=objfun, x0=init, bounds=self.param_bounds, objfun_has_noise=True, **kwargs)