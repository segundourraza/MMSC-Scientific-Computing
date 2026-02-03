import numpy as np

GAUSS_QUADRATURE_POINTS = {1: 0,
                           2: [-1/np.sqrt(3), 1/np.sqrt(3)],
                           3: [-np.sqrt(3/5), 0, np.sqrt(3/5)],
                           4: [-np.sqrt(3/7 + 2/7*np.sqrt(6/5)), -np.sqrt(3/7 - 2/7*np.sqrt(6/5)), np.sqrt(3/7 - 2/7*np.sqrt(6/5)), np.sqrt(3/7 + 2/7*np.sqrt(6/5))],
                           5: [-1/3*np.sqrt(5 + 2*np.sqrt(10/7)), -1/3*np.sqrt(5 - 2*np.sqrt(10/7)), 0, 1/3*np.sqrt(5 - 2*np.sqrt(10/7)), 1/3*np.sqrt(5 + 2*np.sqrt(10/7))]}

GAUSS_QUADRATURE_WEIGHTS = {1: 2, 
                            2: [1,1], 
                            3: [5/9, 8/9, 5/9],
                            4: [(18-np.sqrt(30))/36, (18+np.sqrt(30))/36, (18-np.sqrt(30))/36, (18+np.sqrt(30))/36],
                            5: [(322-13*np.sqrt(70))/900, (322+13*np.sqrt(70))/900, 128/225, (322+13*np.sqrt(70))/900, (322-13*np.sqrt(70))/900]}



def quadrature(f, n):
    I = 0
    for xi, wi in zip(*np.polynomial.legendre.leggauss(n)):
        I += wi*f(xi)
    return I