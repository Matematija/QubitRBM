import numpy as np
from scipy.special import logsumexp
from numba import njit

def hilbert_iter(n_qubits):
    """
    An iterator over all 2**n_qubits bitstrings in a given hilbert space basis.
    """ 
    for n in range(2**n_qubits):
        yield np.fromiter(map(int, np.binary_repr(n, width=n_qubits)), dtype=np.bool, count=n_qubits)

@njit
def log1pexp(z):
    """
    Calculates log(1 + exp(z)) in a numerically stable way.
    """ 
    m = np.max(z.real)
    return m + np.log(np.exp(-m) + np.exp(z-m))

@njit
def logcosh(z):
    """
    Calculates log(cosh(z)) in a numerically stable way.
    """ 
    return -np.log(2) + z + log1pexp(-2*z)

@njit
def sigmoid(z):
    """
    Calculates 1/(1+exp(-z)).
    """ 
    return 0.5*(1 + np.tanh(z/2))

@njit
def logaddexp(x, y, b=(1.0, 1.0)):
    """
    Calculates log(exp(x) +  exp(y)) in a numerically stable way and with numpy broadcasting rules.
    """

    mx = np.max(x.real)
    my = np.max(y.real)
    m = mx if mx > my else my

    return m + np.log(b[0]*np.exp(x-m) + b[1]*np.exp(y-m))

def fold_imag(z):
    """
    Returns z with imaginary values translated to the [-pi,pi] range. (If z is interpreted as a principal-branch logarithm.)
    """
    return np.real(z) + 1j*((np.imag(z) + np.pi)%(2*np.pi) - np.pi)

def exact_fidelity(psi, phi):
    """
    Calculates the mod square overlap of two normalized state vectors.
    """
    return np.abs(np.vdot(psi, phi))**2

def mcmc_fidelity(psipsi, psiphi, phipsi, phiphi):
    """
    Calculates the fidelity estimator between two quantum states represented by their samples.
    The two states are labeled by psi and phi and their bitstring samples by psi_samples and phi_samples, respectively.

    psipsi: numpy.array
        Represents log-values of the psi wavefunction evaluated on psi_samples -> log(psi(psi_samples)).

    psiphi: numpy.array
        Represents log-values of the psi wavefunction evaluated on phi_samples -> log(psi(phi_samples)).

    phipsi: numpy.array
        Represents log-values of the phi wavefunction evaluated on psi_samples -> log(phi(psi_samples)).

    phiphi: numpy.array
        Represents log-values of the phi wavefunction evaluated on phi_samples -> log(phi(phi_samples)).

    Returns: A float representing the overlap squared.
    """


    term_1 = logsumexp(phipsi - psipsi, b=1/phipsi.shape[0])
    term_2 = logsumexp(psiphi - phiphi, b=1/psiphi.shape[0])

    return np.exp(term_1 + term_2).real

def bootstrap_cost_error(vals):
    """
    A convenience function for error estimation from Markov Chains. (Probably needs to get rewritten.)
    """
    n = vals.shape[0]
    return np.random.choice(vals, size=(n,n), replace=True).mean(axis=1).std()