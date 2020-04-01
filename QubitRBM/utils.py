import numpy as np
from scipy.special import logsumexp

def sigmoid(z):
    return 1/(1+np.exp(-z))

def log1pexp(Z, keepdims=False):
    
    z = np.atleast_1d(Z)
    
    x = np.real(z)
    y = np.imag(z)
    
    cond = x < 700
    notcond = np.logical_not(cond)
    
    small = np.zeros(shape=z.shape, dtype=z.dtype)
    big = np.zeros(shape=z.shape, dtype=z.dtype)
    
    np.log1p(np.exp(z, where=cond), out=small, where=cond)
    np.add(x, np.log(np.exp(-x, where=notcond) + np.exp(1j*y, where=notcond), where=notcond), out=big, where=notcond)
    
    res = np.where(cond, small, big) 
    
    return res if res.size > 1 or keepdims else res.item()

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
    term_1 = logsumexp(phipsi - psipsi) - np.log(len(psipsi))
    term_2 = logsumexp(psiphi - phiphi) - np.log(len(phiphi))

    return np.exp(term_1 + term_2).real