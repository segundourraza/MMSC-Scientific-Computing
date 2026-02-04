import numpy as np

def quadrature(f, n):
    I = 0
    for xi, wi in zip(*np.polynomial.legendre.leggauss(n)):
        I += wi*f(xi)
    return I