import numpy as np
from scipy.special import logsumexp

def _bit(x, k, n):
    """
    Returns the k-th binary digit of number x, padded with zeros to length n.
    """
    return int(bin(x)[2:].rjust(n,'0')[k])

def hilbert_iter(n_qubits):
    for n in range(2**n_qubits):
        yield np.fromiter(map(int, np.binary_repr(n, width=n_qubits)), dtype=np.bool, count=n_qubits)

def sigmoid(z):

    big = z.real >= 0
    small = np.logical_not(big)

    em = np.exp(-z, where=big)
    ep = np.exp(z, where=small)

    return np.where(big, 1/(1+em), ep/(1+ep))

def fold_imag(z):
    if z.size > 1:
        cond = np.abs(z.imag) > np.pi
        new_imag = z.imag.copy()
        new_imag[cond] = ((new_imag[cond] + np.pi)%(2*np.pi) - np.pi)
        return z.real + 1j*new_imag
    else:
        return z.real + 1j*((z.imag + np.pi)%(2*np.pi) - np.pi)

def log1pexp(z):   
    m = np.max(z.real)
    return m + np.log(np.exp(-m) + np.exp(z-m))

def pack_params(a, b, W):
    return np.concatenate([a, b, W.reshape(-1)])
    
def unpack_params(params, nv, nh):
    a = params[:nv]
    b = params[nv:(nv+nh)]
    W = params[(nv+nh):].reshape(nv, nh)
    
    return a, b, W

def exact_fidelity(psi, phi):
    return np.abs(np.vdot(psi, phi))**2

def mcmc_fidelity(psipsi, psiphi, phipsi, phiphi):
    term_1 = logsumexp(phipsi - psipsi, b=1/phipsi.shape[0])
    term_2 = logsumexp(psiphi - phiphi, b=1/psiphi.shape[0])

    return np.exp(term_1 + term_2).real