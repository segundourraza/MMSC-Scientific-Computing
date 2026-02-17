import matplotlib.pyplot as plt
import numpy as np
from fem.solver1D import CahnHilliardSolver1D
# np.set_printoptions(suppress=True)
# np.set_printoptions(precision=4,suppress=True)
np.set_printoptions(linewidth = 240)


if __name__ == '__main__':

    ############################################################
    # TIME STEPPERS AND APPRORPIATE PARAMETERS

    # Explicit
    dt = 1e-8
    time_integrator = 0

    # Implicit
    dt = 1e-6
    time_integrator = 1
    
    # Semi implicit CASE A
    dt = 1e-4
    time_integrator = 2


    # Semi implicit CASE B
    dt = 1e-4
    time_integrator = 3


    # # 1ST ORDER SEMI IMPLICIT SCHEME
    # dt = 1e-6
    # time_integrator = '1si'

    # 1ST ORDER STABILIZED SEMI IMPLICIT SCHEME
    dt = 1e-5
    time_integrator = '1ssi'

    
    # 2ND ORDER STABILIZED SEMI IMPLICIT SCHEME
    dt = 1e-5
    time_integrator = '2ssi'


    ###################################################
    # General Parameters
    epsilon = 0.01
    L = 1
    poly_degree = 1
    
    # Initial conditions
    def c0(x): return np.cos(np.pi/L*x)
    
    # T
    tEnd = dt*1
    tEnd = 1e-3
    # tEnd = 5e-3
    

    # for N in [4, 10, 50, 100, 500, 1000, 5000]:
    #   v  # Nonlinear solver
    #     sol = CahnHilliardSolver1D(epsilon, 
    #                             number_of_elements=N, L = L, 
    #                             polynomial_order=poly_degree)
        
    #     nonlinear_solver_options = {'run_checks': False,
    #                         #  'line_search': 'armijo',
    #                         #  'relaxation_parameter': 0.8,
    #                         'verbose' : False,
    #                         }
    #     sol.solve(c0, tEnd, dt, time_integrator=time_integrator, nonlinear_solver_options=nonlinear_solver_options)
    #     sol.save()


    N = 100
    sol = CahnHilliardSolver1D(epsilon, 
                                number_of_elements=N, L = L, 
                                polynomial_order=poly_degree)        
    nonlinear_solver_options = {'run_checks': False,
                        #  'line_search': 'armijo',
                        #  'relaxation_parameter': 0.8,
                        'verbose' : False,
                        }
    dts = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]    
    for dt in dts:
        sol.solve(c0, tEnd, dt, time_integrator=time_integrator, nonlinear_solver_options=nonlinear_solver_options)
        sol.save()

    ###############################################################
    #  PLOTTING
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
    