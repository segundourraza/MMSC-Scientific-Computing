import matplotlib.pyplot as plt
import numpy as np
from fem.solver import CahnHilliardSolver

if __name__ == '__main__':

    # Physics
    epsilon = 0.01

    # Discretization
    L = 1
    N = 10
    poly_degree = 1
    
    x = np.linspace(0, L, N+1)
    he = x[1] - x[0]
    dt = he**4/10
    tEnd = dt*2
    tEnd = dt*1000

    # Initial conditions
    def c0(x): return np.cos(np.pi*x)
    # def c0(x): return np.sin(np.pi/(L)*x)

    # Nonlinear solver


    sol = CahnHilliardSolver(epsilon, 
                             number_of_elements=N, L = L, 
                             polynomial_order=1)
    sol_U = sol.solve_transient(c0, tEnd, dt)
    
    
    fig1, ax1 = plt.subplots(1,2)
    ax1[0].plot(sol.x, sol_U[0,:sol.N])
    ax1[1].plot(sol.x, sol_U[0,sol.N:])
    for i in range(1,len(sol_U)):
        ax1[0].plot(sol.x, sol_U[i,:sol.N])
        ax1[1].plot(sol.x, sol_U[i,sol.N:])
    ax1[0].set_title('c(x)')
    ax1[1].set_title('w(x)')
    fig1.tight_layout()
    

    fig2, ax2 = plt.subplots(1,2)
    ax2[0].plot(sol.t, sol.mass)
    ax2[1].plot(sol.t, sol.J)
    
    ax2[0].set_title('$\\mathcal{M}(C)$')
    ax2[1].set_title('$\\mathcal{J}(C)$')
    
    plt.show()
    