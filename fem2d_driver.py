import matplotlib.pyplot as plt
import numpy as np
from fem.solver2D import CahnHilliardSolver2D
# np.set_printoptions(suppress=True)
# np.set_printoptions(precision=4,suppress=True)
np.set_printoptions(linewidth = 240)


if __name__ == '__main__':

    ############################################################
    # TIME STEPPERS AND APPRORPIATE PARAMETERS

    # Semi implicit CASE B
    dt = 1e-4
    time_integrator = 3


    # # 1ST ORDER SI SCHEME
    # dt = 1e-6
    # time_integrator = '1si'

    # # 1ST ORDER STABILIZED SI SCHEME
    # dt = 1e-5
    # time_integrator = '1ssi'

    
    # # 2ND ORDER STABILIZED SI SCHEME
    # dt = 1e-4
    # time_integrator = '2ssi'

    dt = 1e-6
    ###################################################
    # General Parameters
    epsilon = 0.01
    
    tEnd = dt*100
    # tEnd = 1e-3
    # tEnd = 5e-3
    a = b = 1


    def c0(x,y): return np.cos(np.pi/a*x)*np.sin(np.pi/b*y)
    def c0(x,y): return np.random.random((len(x),))*2 - 1.0

    # Nonlinear solver
    sol = CahnHilliardSolver2D.rectangular_domain(epsilon, a, b, mesh_size=0.02)
    sol.solve(c0, tEnd, dt, time_integrator=time_integrator)





    #######################################################
    # PLOTTING

    fig, ax = plt.subplots(1,2, constrained_layout = True)
    tcf1 = sol.plot_solution(sol.sol_c[0],  ax= ax[0])
    tcf2 = sol.plot_solution(sol.sol_c[-1], ax= ax[1])

    cbar3 = fig.colorbar(
        tcf2,
        ax=ax,
        orientation="horizontal",
        fraction=0.05,   # thickness of colorbar
        # pad=0.15         # distance from subplots
    )
    cbar3.set_ticks(np.linspace(-1,1, 5))  # fewer ticks → more spacing
    
    # [_.set_aspect(1) for _ in ax]
    # fig.tight_layout()

    fig2, ax2 = plt.subplots(1,2)
    ax2[0].plot(sol.t, abs(sol.mass - sol.mass[0]))
    ax2[1].plot(sol.t, sol.E)

    ax2[0].set_title('$\\mathcal{M}(C)$')
    ax2[0].set_title('$|\\mathcal{M}^n - \\mathcal{M}^0|$')
    ax2[1].set_title('$\\mathcal{J}(C)$')

    ax2[0].set_yscale('log')
    [_.grid() for _ in ax2]
    [_.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0)) for _ in ax2]
    [_.set_xlabel('t') for _ in ax2]
    fig2.tight_layout()



    sol.animate_solution()

    plt.show()