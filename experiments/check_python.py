import numpy as np
import scipy.integrate

r = 0.02
q = 0.01
s = 100.0
k = 100.0
v = 0.04
kappa = 1.0
theta = 0.04
sigma_v = 0.2
rho = -0.7
tau = 1.0

def phi(u, tau):
    alpha_hat = -0.5 * u * (u + 1j)
    beta = kappa - 1j * u * sigma_v * rho
    gamma = 0.5 * sigma_v ** 2
    d = np.sqrt(beta**2 - 4 * alpha_hat * gamma)
    g = (beta - d) / (beta + d)
    h = np.exp(-d*tau)
    A_ = (beta-d)*tau - 2*np.log((g*h-1) / (g-1))
    A = kappa * theta / (sigma_v**2) * A_
    B = (beta - d) / (sigma_v**2) * (1 - h) / (1 - g*h)
    return np.exp(A + B * v)

def integral(k, tau):
    integrand = (lambda u: 
        np.real(np.exp((1j*u + 0.5)*k)*phi(u - 0.5j, tau))/(u**2 + 0.25))

    i, err = scipy.integrate.quad_vec(integrand, 0, np.inf)
    return i

def call(k, tau):
    a = np.log(s/k) + (r-q)*tau
    i = integral(a, tau)        
    return s * np.exp(-q*tau) - k * np.exp(-r*tau)/np.pi*i
    
call(k, tau)