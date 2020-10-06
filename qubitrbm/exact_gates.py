import numpy as np
from scipy import sparse
from functools import reduce
from math import floor

import sys, os

libpath = os.path.abspath('..')
if libpath not in sys.path:
    sys.path.append(libpath)

from QubitRBM.utils import _bit

def _extend_to_n_qubits(matrix, ns, n_qubits):
    I = sparse.eye(2)
    qubits = np.ravel(ns)
    return reduce(sparse.kron, (matrix if i in qubits else I for i in range(n_qubits)))

def X(n=0, n_qubits=1):
    x = np.array([[0, 1], [1, 0]], dtype=np.complex)
    return _extend_to_n_qubits(x, n, n_qubits)

def Y(n=0, n_qubits=1):
    y = np.array([[0, -1j], [1j, 0]], dtype=np.complex)
    return _extend_to_n_qubits(y, n, n_qubits)

def Z(n=0, n_qubits=1):
    z = np.array([[1, 0], [0, -1]], dtype=np.complex)
    return _extend_to_n_qubits(z, n, n_qubits)

def H(n=0, n_qubits=1):
    h = np.array([[1, 1], [1, -1]], dtype=np.complex)/np.sqrt(2)
    return _extend_to_n_qubits(h, n, n_qubits)

def RZ(phi, n=0, n_qubits=1):
    rz = np.array([[1, 0],[0, np.exp(1j*phi)]], dtype=np.complex)
    return _extend_to_n_qubits(rz, n, n_qubits)

def P(phi, n=0, n_qubits=1):
    p = np.array([[0, 1],[np.exp(1j*phi), 0]], dtype=np.complex)
    return _extend_to_n_qubits(p, n, n_qubits)

def RX(phi, n=0, n_qubits=1):
    c = np.cos(phi)
    s = np.sin(phi)
    rx = np.array([[c, -1j*s],[-1j*s, c]], dtype=np.complex)
    return _extend_to_n_qubits(rx, n, n_qubits)

def CRZ(phi, k, l, n_qubits):
    # Make more efficient if it ever becomes necessary.

    g = np.exp(1j*phi)
    diag = np.exp(1j*phi/2)*np.fromiter((g if _bit(j, k, n_qubits) and _bit(j, l, n_qubits) else 1 for j in range(2**n_qubits)), dtype=np.complex, count=2**n_qubits)

    return sparse.diags(diag, dtype=np.complex)

def RZZ(phi, k, l, n_qubits):
    # Make more efficient if it ever becomes necessary.

    g = np.exp(1j*phi)
    diag = np.fromiter((g if _bit(j, k, n_qubits) != _bit(j, l, n_qubits) else 1 for j in range(2**n_qubits)), dtype=np.complex, count=2**n_qubits)

    return sparse.diags(diag, dtype=np.complex)