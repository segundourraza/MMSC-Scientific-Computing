from fem import CahnHilliardSolver2D, CahnHilliardSolver1D
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import bmat
from scipy.linalg import issymmetric

if __name__ == '__main__':

    epsilon = 0.01
    mesh_size = 0.1
    a = b = 1
    dt = 1e-5

    n = 4
    def c0(x,y): return np.cos(np.pi/a*x)*np.sin(np.pi/(b*n)*y)
    def c0(x,y): return np.cos(2*np.pi/(a)*n*x)*np.sin(2*np.pi/(b)*n*y)
    
    # np.random.default_rng(0)
    # def c0(x,y): return np.random.uniform(-1.0, 1, (len(x),))

    # Nonlinear solver

    ##########################################################################
    # 1D Problem
    Ne = 10
    poly_degree = 2
    sol = CahnHilliardSolver1D(epsilon, Ne, b, polynomial_order=poly_degree)


    # sol = CahnHilliardSolver2D.rectangular_domain(epsilon, a, b, mesh_size=mesh_size)
    # sol.plot_mesh()

    
    # Used for Semi-Implicit methods B
    ck = -sol.epsilon*sol.K - (2/sol.epsilon)*sol.M
    # Compute LHS: This is done once for stencil B
    A_eyre = bmat([[ck,     sol.M],
                   [sol.M,  dt*sol.K]], 
                format= 'bsr')
    print(type(A_eyre))
    print(A_eyre.getformat())
    A_eyre = A_eyre.toarray()
    
    print(issymmetric(A_eyre))

    fig = plt.figure()
    plt.spy(A_eyre)

    plt.show()