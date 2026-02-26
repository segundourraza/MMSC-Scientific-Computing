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
    dt = 1e-5
    time_integrator = 3


    # # 1ST ORDER SI SCHEME
    # dt = 1e-6
    # time_integrator = '1si'

    # 1ST ORDER STABILIZED SI SCHEME
    dt = 1e-4
    dt = 1e-3
    time_integrator = '1ssi'

    
    # # 2ND ORDER STABILIZED SI SCHEME
    # dt = 1e-4
    # dt = 1e-3
    # time_integrator = '2ssi'

    # dt = 1e-6


    ###################################################
    # General Parameters
    epsilon = 0.01
    
    tEnd = dt*1
    # tEnd = 1e-3
    # tEnd = 1e-1
    tEnd = 1
    a = b = 1

    def c0(x,y): return np.cos(np.pi/a*x)*np.sin(np.pi/(b)*y)

    n = 4
    def c0(x,y): return np.cos(np.pi/np.sqrt(a**2 + b**2)*n*(x + y)/np.sqrt(2))

    n = 4
    def c0(x,y): return np.cos(np.pi/np.sqrt(a**2 + b**2)*n*(x + y)/np.sqrt(2))*np.sin(np.pi/np.sqrt(a**2 + b**2)*n*(x - y)/np.sqrt(2))

    # n = 6
    # def c0(x,y): return np.cos(2*np.pi/(a)*n*x)*np.sin(2*np.pi/(b)*n*y)
    

    # np.random.default_rng(0)
    # def c0(x,y): return np.random.uniform(-1.0, 1, (len(x),))

    prepend = None
    save = True

    mesh_size = 0.02
    tris = CahnHilliardSolver2D.rectangular_domain_tri(epsilon, a, b, mesh_size=mesh_size)

    
    nx = 20
    ny = 20
    order = 2
    quads = CahnHilliardSolver2D.rectangular_domain_rect(epsilon, a, b, nx, ny, order)

    tris.solve(c0, tEnd, dt, time_integrator=time_integrator)
    quads.solve(c0, tEnd, dt, time_integrator=time_integrator)
    
    if save:    
        quads.save(prepend=prepend)
        quads.animate_solution(vector = 'c', fps = 10, show_mesh=True, prepend='quads')
        tris.save(prepend=prepend)
        tris.animate_solution(vector = 'c', fps = 10, show_mesh=True, prepend='tris')


    #######################################################
    # PLOTTING
    
    fig, ax = plt.subplots(2,2, constrained_layout = True, sharex=True, sharey = True)
    ax = ax.flatten()
    vmin = min(np.nanmin(tris.sol_c[-1]),np.nanmin(quads.sol_c[-1]), -1.0)
    vmax = max(np.nanmax(tris.sol_c[-1]), np.nanmin(quads.sol_c[-1]), 1.0)
        
    tcf1, _ = tris.plot_solution(tris.sol_c[0],  ax= ax[0], vmin = vmin, vmax = vmax, plot_mesh=True)
    tcf1, _ = tris.plot_solution(tris.sol_c[-1],  ax= ax[2], vmin = vmin, vmax = vmax, plot_mesh=True)
    
    tcf2, _ = quads.plot_solution(quads.sol_c[0], ax= ax[1], vmin = vmin, vmax = vmax, plot_mesh=True)
    tcf2, _ = quads.plot_solution(quads.sol_c[-1], ax= ax[3], vmin = vmin, vmax = vmax, plot_mesh=True)
    
    ax[0].set_title('Tris: start')
    ax[2].set_title('Tris: End')
    ax[1].set_title('Quads: start')
    ax[3].set_title('Quads: end')
    cbar = fig.colorbar(
        tcf2,
        ax=ax,
        orientation="horizontal",
        fraction=0.05,   # thickness of colorbar
        # pad=0.15         # distance from subplots
    )
    ax[0].set_ylabel('y')
    ax[2].set_ylabel('y')
    ax[2].set_xlabel('x')
    ax[3].set_xlabel('x')






    fig2, ax2 = plt.subplots(1,2)
    for sol in [tris, quads]:
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