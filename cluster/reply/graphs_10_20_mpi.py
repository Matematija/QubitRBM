from mpi4py import MPI

from collections import OrderedDict
from copy import deepcopy
import numpy as np
import networkx as nx
import os, sys

libpath = os.path.abspath(os.path.join('..','..'))
if libpath not in sys.path:
    sys.path.append(libpath)

from qubitrbm.qaoa import QAOA
from qubitrbm.optimize import Optimizer
from qubitrbm.rbm import RBM
from qubitrbm.utils import exact_fidelity

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

MIN_QUBITS = 10
N_GRAPHS = 5
MPI_GROUP_TAG = 0 if mpi_rank%2==0 else 1
N = MIN_QUBITS + 2*(mpi_rank//2)
MCMC_PARAMS = OrderedDict(n_steps=8000, n_chains=4, warmup=1500, step=N)
OUTPUT_FILENAME = f'{N}_qubits_{N_GRAPHS}_graphs_process_{MPI_GROUP_TAG+1}_high_mcmc'
COMPRESSION_ATTEMPTS = 10
P = 4

data = OrderedDict(n_qubits=N, **MCMC_PARAMS)
printout_tag = f'Process {mpi_rank}:'

################################################################################

all_Ns = N
all_Ns = comm.gather(all_Ns, root=0)

if mpi_rank==0:
    print(f'Number of MPI processes: {mpi_size}')
    print(f'Max qubit count: {max(all_Ns)}')

for g in range(N_GRAPHS):

    G = nx.random_regular_graph(d=3, n=N)

    while not nx.is_connected(G):
        G = nx.random_regular_graph(d=3, n=N)

    G = nx.relabel_nodes(G, dict(zip(G.nodes, range(len(G)))))
    data[f'graph_{g}'] = nx.to_numpy_array(G, nodelist=sorted(G.nodes), dtype=int)

    qaoa = QAOA(G, p=P)

    print(f'{printout_tag} Starting the initial QAOA angles optimization for graph {g+1}/{N_GRAPHS} at {N} qubits, p={P}', flush=True)

    angles, costs = qaoa.optimize(init=P*[np.pi/8] + P*[-np.pi/8], lr=1e-3, tol=1e-3, dx=1e-7, verbose=False)
    gammas, betas = np.split(angles, 2)

    data[f'graph_{g}_opt_gammas'] = gammas
    data[f'graph_{g}_opt_betas'] = betas
    data[f'graph_{g}_opt_cost'] = costs[-1]

    print(f'{printout_tag} Finished initial QAOA angles optimization for graph {g+1}/{N_GRAPHS} at {N} qubits, p={P}, reached cost={costs[-1]} in {len(costs)} iterations.', flush=True)

    logpsi = RBM(N)
    optim = Optimizer(logpsi, **MCMC_PARAMS)

    for p in range(1,P+1):

        logpsi.UC(G, gammas[p-1])
        optim.machine = logpsi

        if p>1:
            # Compression:

            gamma_init = optim.optimal_compression_init(G)

            aux = RBM(N)
            aux.UC(G, gamma_init, mask=False)
            init_params = deepcopy(aux.params)

            for counter in range(COMPRESSION_ATTEMPTS):
                params, history = optim.sr_compress(init=init_params, lookback=25, resample_phi=2, verbose=False)

                if history[-1] > 0.9:
                    break
            
            if counter == COMPRESSION_ATTEMPTS -1:
                print(f'{printout_tag} COMPRESSION FAILED at p={p}, graph {g+1}/{N_GRAPHS}')

            print(f'{printout_tag} Finished compression at p={p}, graph {g+1}/{N_GRAPHS}, reached fidelity {history[-1]} after {counter+1} attempts', flush=True)

            nh = (len(params) - N)//(N+1)
            logpsi = RBM(N, nh)
            logpsi.params = params
            optim.machine = logpsi

        for n in range(N):
            params, history = optim.sr_rx(n=n, beta=betas[p-1], lr=7e-2, lookback=8, resample_phi=6, eps=1e-4, verbose=False)
            logpsi.params = params
            print(f'{printout_tag} Done with qubit {n+1}/{N} at graph {g+1}/{N_GRAPHS}, depth p={p}/{P}, reached fidelity {history[-1]}', flush=True)

        data[f'params_graph{g}_after_p{p}'] = params

        psi_exact = QAOA(G, p=p).simulate(gammas[:p], betas[:p]).final_state_vector
        psi_rbm = logpsi.get_state_vector(normalized=True)
        F = exact_fidelity(psi_exact, psi_rbm)
        data[f'exact_fidelity_graph{g}_after_p{p}'] = F
        print(f'{printout_tag} Finished with p={p}/{P} for graph {g+1}/{N_GRAPHS}. Reached exact fidelity: {F}', flush=True)

        np.savez(OUTPUT_FILENAME, **data)