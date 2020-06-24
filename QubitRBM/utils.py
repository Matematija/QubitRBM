import numpy as np
from scipy.special import logsumexp
from numba import njit

def _bit(x, k, n):
    """
    Returns the k-th binary digit of number x, padded with zeros to length n.
    """
    return int(bin(x)[2:].rjust(n,'0')[k])

def hilbert_iter(n_qubits):
    for n in range(2**n_qubits):
        yield np.fromiter(map(int, np.binary_repr(n, width=n_qubits)), dtype=np.bool, count=n_qubits)

@njit
def log1pexp(z):   
    m = np.max(z.real)
    return m + np.log(np.exp(-m) + np.exp(z-m))

@njit
def logcosh(z):
    return -np.log(2) + z + log1pexp(-2*z)

@njit
def sigmoid(z):
    return 0.5*(1 + np.tanh(z/2))

@njit
def logaddexp(x, y, b=(1.0, 1.0)):

    mx = np.max(x.real)
    my = np.max(y.real)
    m = mx if mx > my else my

    return m + np.log(b[0]*np.exp(x-m) + b[1]*np.exp(y-m))

def fold_imag(z):
    if z.size > 1:
        cond = np.abs(z.imag) > np.pi
        new_imag = z.imag.copy()
        new_imag[cond] = ((new_imag[cond] + np.pi)%(2*np.pi) - np.pi)
        return z.real + 1j*new_imag
    else:
        return z.real + 1j*((z.imag + np.pi)%(2*np.pi) - np.pi)

def exact_fidelity(psi, phi):
    return np.abs(np.vdot(psi, phi))**2

def mcmc_fidelity(psipsi, psiphi, phipsi, phiphi):
    term_1 = logsumexp(phipsi - psipsi, b=1/phipsi.shape[0])
    term_2 = logsumexp(psiphi - phiphi, b=1/psiphi.shape[0])

    return np.exp(term_1 + term_2).real

@njit
def bootstrap_cost_error(vals):

    n = vals.shape[0]
    bootstrapped = np.zeros_like(vals)

    for i in range(n):
        bootstrapped[i] = np.random.choice(vals, size=n, replace=True).mean()

    return bootstrapped.std()