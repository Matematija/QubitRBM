import numpy as np
import networkx as nx

import cirq
import sympy

import pybobyqa as optim

class QAOA:
    
    def __init__(self, graph):
        
        self.n_qubits = len(graph.nodes)
        
        self.qubits = cirq.LineQubit.range(self.n_qubits)
        self.graph = nx.relabel_nodes(graph, mapping={i: self.qubits[i] for i in range(self.n_qubits)})
        
        self.gamma_param = sympy.Symbol('gamma')
        self.beta_param = sympy.Symbol('beta')
        self.param_bounds = np.array([[0, -np.pi/4], [np.pi/2, np.pi/4]])
        
        self.circuit = cirq.Circuit(
                            cirq.H.on_each(self.graph.nodes()),
                            (cirq.ZZPowGate(exponent=2*self.gamma_param/np.pi)(u, v) for u, v in self.graph.edges()),
                            cirq.Moment(cirq.rx(rads=2*self.beta_param)(qubit) for qubit in self.graph.nodes()),
                            (cirq.measure(qubit) for qubit in self.graph.nodes())
                        )
        
        self.sim = cirq.Simulator(dtype=np.complex128)
    
    def cost(self, samples):

        cost_value = 0.0

        for qi, qj in self.graph.edges():
            term = (-1)**(samples[:,qi.x]*samples[:,qj.x])
            cost_value += np.mean(term)

        return cost_value

    def cost_from_params(self, params, n_samples):
        samples = self.sample(*params, n_samples)
        return self.cost(samples)
    
    def sample(self, gamma, beta, n_samples):
        param_res = cirq.ParamResolver({self.gamma_param: gamma, self.beta_param: beta})
        res = self.sim.sample(program=self.circuit, params=param_res, repetitions=n_samples)
        
        return res.drop(labels=['gamma', 'beta'], axis=1).values
    
    def optimize(self, n_samples, init=None, **kwargs):

        if init is None:
            gamma = np.random.uniform(0, np.pi/2)
            beta = np.random.uniform(-np.pi/4, np.pi/4)
            init = np.array([gamma, beta])

        objfun = lambda params: self.cost_from_params(params, n_samples)

        return optim.solve(objfun=objfun, x0=init, bounds=self.param_bounds, objfun_has_noise=True, **kwargs)

        