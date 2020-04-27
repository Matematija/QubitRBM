#!/usr/bin/env python
# coding: utf-8

import numpy as np

# import qiskit
# from qiskit import Aer

import sys, os

sys.path.append(os.path.abspath('..'))

from QubitRBM.optimize import *
from QubitRBM.rbm import *

nv = 20
nh = 0 # to start off with, we can add more later

depth = 500
gates = [('RZZ', 2), ('RZ', 1), ('P', 1)]

data = {}

logpsi = RBM(nv, nh)

# qubits = qiskit.QuantumRegister(size=nv, name='q')
# circ = qiskit.QuantumCircuit(qubits)
# circ.h(qubits)

chosen_gates = []

for k in range(depth):
    
    i = np.random.randint(low=0, high=len(gates))
    gate, nqs = gates[i]
    chosen_gates.append(gates[i])
    
    qs = np.random.choice(np.arange(nv), size=nqs, replace=False)
    phi = np.random.uniform(low=-np.pi, high=np.pi)
    
    if gate=='RZ':
        logpsi.RZ(*qs, phi)
        # circ.rz(phi, *qs)
    elif gate=='RZZ':
        logpsi.RZZ(*qs, phi)
        # circ.rzz(phi, *qs)
    elif gate=='P':
        logpsi.P(*qs, phi)
        # circ.x(*qs)
        # circ.rz(phi, *qs)

logpsi.fold_imag_params()

data['chosen_gates'] = np.array(chosen_gates)
data['a_pre_h'] = logpsi.a.copy()
data['b_pre_h'] = logpsi.b.copy()
data['W_pre_h'] = logpsi.W.copy()

################# Qiskit exact result #################

# circ.h(qubits)

# backend = Aer.get_backend('statevector_simulator')

# job = qiskit.execute(circ, backend)
# result = job.result()
# psi_exact = result.get_statevector(circ, decimals=50)

# p_exact = np.abs(psi_exact)**2
# print('Qiskit probability sum check: {}'.format(p_exact.sum()))

################# Hadamard transform #################

print("Ratio nh/nv: ", logpsi.nh/logpsi.nv)

r = 8

if logpsi.nh/logpsi.nv < r:
    print("Ratio nh/nv too low, adding to get to {}".format(r))
    logpsi.add_hidden_units(r*logpsi.nv - logpsi.nh)

lr = 1e-1
lrf = 8e-2
# steps = 400
# lr_tau = steps/np.log(lr/lrf)
lr_tau = None
lr_min = lrf

tol = 1e-5

add_units = 5

for n in range(nv):
    
    print('Qubit {} starting...'.format(n+1))
    
    while True:

        a, b, W, Fs = hadamard_optimization(logpsi, n, parallel=True, tol=tol, lr=lr, lr_tau=lr_tau, lr_min=lr_min,
                                           lookback=50, resample_phi=None, sigma=0.0, fidelity='mcmc',
                                           psi_mcmc_params=(50000,2000,1), phi_mcmc_params=(50000,2000,1),
                                           eps=1e-6, verbose=True)
        
        if Fs[-1] > 0.92:
            logpsi.set_params(a=a, b=b, W=W)
            logpsi.fold_imag_params()

            data['qubit_{}_Fs'.format(n)] = Fs

            break
        else:
            print('||Repeating optimization, adding {} hidden units||'.format(add_units))
            logpsi.add_hidden_units(add_units)
    
    print('\nQubit {} done! Final fidelity estimate: {:05.4f}'.format(n+1, Fs[-1]))

data['a_final'] = a
data['b_final'] = b
data['W_final'] = W

np.savez('weights', **data)