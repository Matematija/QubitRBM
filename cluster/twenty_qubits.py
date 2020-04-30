#!/usr/bin/env python

import numpy as np
import sys, os

sys.path.append(os.path.abspath('..'))

from QubitRBM.optimize import *
from QubitRBM.rbm import *

from mpi4py import MPI

comm = MPI.COMM_WORLD
r = comm.Get_rank()
p = comm.Get_size()

nv = 20
nh = 1 # to start off with, we can add more later

if r == 0:

    print('Numpy config:\n')
    np.__config__.show()
    print('\n')

    logpsi = RBM(nv, nh)

    depth = 500
    gates = [('RZZ', 2), ('RZ', 1), ('P', 1)]
    data = {}
    chosen_gates = []

    for k in range(depth):
        
        i = np.random.randint(low=0, high=len(gates))
        gate, nqs = gates[i]
        chosen_gates.append(gates[i])
        
        qs = np.random.choice(np.arange(nv), size=nqs, replace=False)
        phi = np.random.uniform(low=-np.pi, high=np.pi)
        
        if gate=='RZ':
            logpsi.RZ(*qs, phi=phi)
        elif gate=='RZZ':
            logpsi.RZZ(*qs, phi=phi)
        elif gate=='P':
            logpsi.P(*qs, phi=phi)

    logpsi.fold_imag_params()

    data['chosen_gates'] = np.array(chosen_gates)
    data['a_pre_h'] = logpsi.a.copy()
    data['b_pre_h'] = logpsi.b.copy()
    data['W_pre_h'] = logpsi.W.copy()

    print("Ratio nh/nv: ", logpsi.nh/logpsi.nv)

    ratio = 8

    if logpsi.nh/logpsi.nv < ratio:
        print("Ratio nh/nv too low, adding to get to {}".format(ratio))
        logpsi.add_hidden_units(ratio*logpsi.nv - logpsi.nh)

else:
    logpsi = None

logpsi = comm.bcast(logpsi, root=0)

########## Parallel optimization ##########

lr = 1e-1
lrf = 8e-2
# steps = 400
# lr_tau = steps/np.log(lr/lrf)
lr_tau = None
lr_min = lrf

tol = 1e-5
add_units = 2

for n in range(nv):
    
    if r==0:
        print('Qubit {} starting...'.format(n+1))
    
    while True:

        # a, b, W, Fs = parallel_hadamard_optimization(logpsi, comm, n, tol=tol, lr=lr, lr_tau=lr_tau, lr_min=lr_min,
        #                                                 lookback=50, resample_phi=None, sigma=0.0,
        #                                                 psi_mcmc_params=(20000,50,1), phi_mcmc_params=(20000,50,1),
        #                                                 eps=1e-6, verbose=True)

        a, b, W, Fs = parallel_hadamard_optimization_2(logpsi, comm, n, tol=tol, lr=lr,
                                                        lookback=50, resample_phi=None, sigma=0.0,
                                                        psi_mcmc_params=(2000,100,10), phi_mcmc_params=(2000,100,10),
                                                        eps=1e-6, verbose=True)
        
        if Fs[-1] > 0.92:
            logpsi.set_params(a=a, b=b, W=W)
            logpsi.fold_imag_params()

            if r==0:
                data['qubit_{}_Fs'.format(n)] = Fs

            break
        else:
            if r==0:
                print('||Repeating optimization, adding {} hidden units||'.format(add_units))
            logpsi.add_hidden_units(add_units)

    if r==0:
        print('\nQubit {} done! Final fidelity estimate: {:05.4f}'.format(n+1, Fs[-1]))

if r==0:
    data['a_final'] = a
    data['b_final'] = b
    data['W_final'] = W

    np.savez('weights', **data)