import numpy as np
from scipy.special import logsumexp

def sigmoid(z):

    em = np.exp(-z, where=z.real>=0)
    ep = np.exp(z, where=z.real<0)

    return np.where(z.real>=0, 1/(1+em), ep/(1+ep))

def log1pexp(z, keepdims=False):
    
    pad = np.zeros_like(z)
    stacked = np.stack([pad, z], axis=0)
    
    return logsumexp(stacked, axis=0, keepdims=keepdims)

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