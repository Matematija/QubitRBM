import numpy as np

def logsumexp(arr, axis=None):
    m = np.max(arr, axis=axis, keepdims=True)
    return m.squeeze() + np.log(np.exp(arr - m).sum(axis=axis))

def logmeanexp(arr, axis=None, keepdims=False):
    m = np.max(arr, axis=axis, keepdims=True)
    res = m + np.log(np.exp(arr - m).mean(axis=axis, keepdims=True))
    return res.squeeze() if not keepdims else res

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
    
    pad = np.zeros_like(z)
    stacked = np.stack([pad, z], axis=0)
    
    return logsumexp(stacked, axis=0)

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
    term_1 = logmeanexp(phipsi - psipsi)
    term_2 = logmeanexp(psiphi - phiphi)

    return np.exp(term_1 + term_2).real