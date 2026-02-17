import matplotlib.pyplot as plt
import numpy as np
from fem.solver2D import CahnHilliardSolver2D
# np.set_printoptions(suppress=True)
# np.set_printoptions(precision=4,suppress=True)
np.set_printoptions(linewidth = 240)


if __name__ == '__main__':

    ############################################################
    # TIME STEPPERS AND APPROPRIATE PARAMETERS

    # Semi implicit CASE B
    mesh_size = 0.02
    dt = 1e-5
    time_integrator = 3


    # # 1ST ORDER SI SCHEME
    # dt = 1e-6
    # time_integrator = '1si'

    # 1ST ORDER STABILIZED SI SCHEME
    dt = 1e-5
    mesh_size = 0.02
    time_integrator = '1ssi'

    
    # # 2ND ORDER STABILIZED SI SCHEME
    # dt = 1e-4
    # time_integrator = '2ssi'

    # dt = 1e-6
    ###################################################
    # General Parameters
    epsilon = 0.01
    
    # tEnd = dt*10
    tEnd = 1e-3
    # tEnd = 5e-3
    a = b = 1

    n = 4
    def c0(x,y): return np.cos(np.pi/a*x)*np.sin(np.pi/(b*n)*y)
    def c0(x,y): return np.cos(2*np.pi/(a)*n*x)*np.sin(2*np.pi/(b)*n*y)
    
    # np.random.default_rng(0)
    # def c0(x,y): return np.random.uniform(-1.0, 1, (len(x),))

    # Nonlinear solver


    sol = CahnHilliardSolver2D.rectangular_domain(epsilon, a, b, mesh_size=mesh_size)
    # sol.plot_mesh()
    
    sol.solve(c0, tEnd, dt, time_integrator=time_integrator)
    
    sol.save()
    
    sol.animate_solution(vector = 'c', fps = 10)
    # sol.animate_solution(vector = 'w', fps = 10)


    #######################################################
    # PLOTTING
    data = np.zeros((sol.nt,2))
    for it in range(sol.nt):
        data[it,:] = [np.nanmin(sol.sol_c[it]), np.nanmax(sol.sol_c[it])]

    fig, ax = plt.subplots(1,2)
    ax[0].plot(sol.t, data[:,0], label = "max(c)")
    ax[1].plot(sol.t, data[:,1], label = "min(c)")

    
    fig, ax = plt.subplots(1,2, constrained_layout = True)
    vmin = min(np.nanmin(sol.sol_c[-1]), -1.0)
    vmax = max(np.nanmax(sol.sol_c[-1]), 1.0)
        
    tcf1, _ = sol.plot_solution(sol.sol_c[0],  ax= ax[0], vmin = vmin, vmax = vmax)
    tcf2, _ = sol.plot_solution(sol.sol_c[-1], ax= ax[1], vmin = vmin, vmax = vmax)
    ax[0].set_title('Starting solution')
    ax[1].set_title('Ending solution')

    cbar = fig.colorbar(
        tcf2,
        ax=ax,
        orientation="horizontal",
        fraction=0.05,   # thickness of colorbar
        # pad=0.15         # distance from subplots
    )
    # cbar.set_ticks(np.linspace(-1,1, 5))  # fewer ticks → more spacing
    
    
    # fig, ax = plt.subplots(1,2, constrained_layout = True)
    # tcf1, levels1 = sol.plot_solution(sol.sol_w[0],  ax= ax[0])
    # tcf2, levels2 = sol.plot_solution(sol.sol_w[-1], ax= ax[1])

    # cbar = fig.colorbar(
    #     tcf2,
    #     ax=ax,
    #     orientation="horizontal",
    #     fraction=0.05,   # thickness of colorbar
    #     # pad=0.15         # distance from subplots
    # )
    # cbar.set_ticks(np.linspace(levels2[0],levels2[-1], 5))  # fewer ticks → more spacing
    






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




    plt.show()