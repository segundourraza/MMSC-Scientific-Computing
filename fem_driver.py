import matplotlib.pyplot as plt
import numpy as np
from fem.solver import CahnHilliardSolver
np.set_printoptions(suppress=True)
# np.set_printoptions(precision=4,suppress=True)
np.set_printoptions(linewidth = 240)


if __name__ == '__main__':

    # Physics
    epsilon = 0.01

    # Discretization
    L = 1.5
    N = 50
    poly_degree = 2
    
    # Explicit
    dt = 1e-8

    # Implicit
    dt = 1e-4

    tEnd = dt*10
    # # tEnd = dt*0
    # tEnd = 1e-3
    tEnd = 5e-3
    time_integrator = 2
    
    
    print(int(tEnd/dt))


    # Initial conditions
    def c0(x): return np.cos(np.pi/L*x)
    # def c0(x): return np.sin(np.pi/(L)*x)

    # Nonlinear solver


    sol = CahnHilliardSolver(epsilon, 
                             number_of_elements=N, L = L, 
                             polynomial_order=poly_degree)
    sol.solve(c0, tEnd, dt, time_integrator=time_integrator)

    sol.plot_contour()
    
    fig1, ax1 = plt.subplots(1,2)
    ax1[0].plot(sol.x, sol.sol_c[0], '.-')
    ax1[1].plot(sol.x, sol.sol_w[0], '.-')
    if len(sol.t) > 1:
        for i in range(1,len(sol.t), max(len(sol.t)//10, 1)):
            ax1[0].plot(sol.x, sol.sol_c[i])
            ax1[1].plot(sol.x, sol.sol_w[i])
    ax1[0].set_title('c(x)')
    ax1[1].set_title('w(x)')
    fig1.tight_layout()
    [_.set_xlabel('x') for _ in ax1]
    [_.grid() for _ in ax1]
    

    fig2, ax2 = plt.subplots(1,2)
    ax2[0].semilogy(sol.t[1:], abs(sol.mass[1:]- sol.mass[0]))
    ax2[1].plot(sol.t, sol.J)
    
    ax2[0].set_title('$\\mathcal{M}(C)$')
    ax2[0].set_title('$|\\mathcal{M}^n - \\mathcal{M}^0|$')
    ax2[1].set_title('$\\mathcal{J}(C)$')

    [_.grid() for _ in ax2]
    [_.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0)) for _ in ax2]
    [_.set_xlabel('t') for _ in ax2]
    fig2.tight_layout()

    plt.show()
    