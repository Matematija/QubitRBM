import os, sys
assert len(sys.argv) == 3, 'Too few arguments given!'

import numpy as np
import networkx as nx

import qiskit
from qiskit.providers.aer import QasmSimulator

path = os.path.abspath('../../')
sys.path.append(path)

print(f'Looking for QubitRBM in {path}')

from qubitrbm.qaoa import QAOA

edgelist_path = sys.argv[1]
G = nx.read_edgelist(edgelist_path, data=False, nodetype=int)

N = len(G.nodes)
k = G.degree[0]

print(f'Read a {k}-regular graph with {N} nodes', flush=True)

angles_path = sys.argv[2]
gammas, betas = np.split(np.loadtxt(angles_path), 2)

p = len(gammas)
print(f'p = {p}', flush=True)

NDIMS = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
PROC = int(os.environ["SLURM_ARRAY_TASK_ID"])
MAXDIM = 2700
MINDIM = 50
DIM = MINDIM + PROC*((MAXDIM-MINDIM)//NDIMS)
NSAMPLES = 20000

print(f'Read in QAOA angle data from {angles_path}')
print(f'Gammas = {gammas}')
print(f'Betas = {betas}')
print(f'MAXDIM = {MAXDIM}')
print(f'MINDIM = {MINDIM}')
print(f'PROC = {PROC}')
print(f'DIM = {DIM}')
print(f'NSAMPLES = {NSAMPLES}', flush=True)

print('Defining the circuit...')

circ = qiskit.QuantumCircuit(N)

circ.h(range(N))

for gamma, beta in zip(gammas, betas):

    for i, j in G.edges():
        circ.rzz(2*gamma, i, j)
        
    circ.barrier()
    circ.rx(2*beta, range(N))
    
circ.measure_all()

backend = QasmSimulator()

backend_options = {"method": "matrix_product_state",
                   "matrix_product_state_max_bond_dimension": DIM,
                   "matrix_product_state_truncation_threshold": 1e-15}

job = qiskit.execute(circ, backend, backend_options=backend_options, shots=NSAMPLES)
result = job.result()

print('Finished sim, getting counts...', flush=True)

counts = result.get_counts(circ)

bitstrings = np.zeros(shape=[NSAMPLES, N], dtype=int)
probs = np.zeros(shape=[NSAMPLES], dtype=float)

for i, (s, count) in enumerate(counts.items()):
    bitstrings[i,:] = [int(b) for b in s[::-1]]
    probs[i] = count/NSAMPLES

save_path = f'qiskit_proc_{PROC}_data.npz'

print(f'Saving data to {save_path}', flush=True)
np.savez(save_path, bitstrings=bitstrings, probs=probs)

print('Calculating cost:', flush=True)

qaoa = QAOA(G, p)
mean_cost = np.sum(probs*qaoa.cost(bitstrings))

print(f'Mean cost: {mean_cost}')