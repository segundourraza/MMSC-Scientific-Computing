import h5py
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({
    "mathtext.fontset": "cm",   # Computer Modern
    "font.family": "serif"
})

import numpy as np


def load_solution_hdf5(filepath):
    arrays = {}
    scalars = {}

    with h5py.File(filepath, "r") as f:
        sol_grp = f["solution"]

        # Load arrays
        for name, ds in sol_grp["arrays"].items():
            arrays[name] = ds[:]

        # Load scalars
        for name, ds in sol_grp["scalars"].items():
            scalars[name] = ds[()]


    return arrays, scalars

    
def analyzer_contours(file_name, nt, plot_mesh = False, levels = 100, cmap = 'jet', fp = Path.cwd() / "solution2D"):

    
    # COMPARISON OF 1st ORDER STABILIZED SEMI IMPLICIT METHOD    
    arrays, scalars = load_solution_hdf5(fp / file_name)
    
    fig, ax = plt.subplots(1,2)
    ax[0].semilogy(arrays['t'], abs(arrays['mass'] - arrays['mass'][0]))
    ax[1].plot(arrays['t'], arrays['energy'])
    
        
    ax[1].legend(fontsize = 12)
    ax[0].set_title("Error in Mass")
    ax[0].set_ylabel('$|\\mathcal{M}(c^n) - \\mathcal{M}(c^0)|$')
    ax[1].set_title("Ginzburg-Landau Energy")
    ax[1].set_ylabel('$\\mathcal{E}(c)$')
    
    [_.grid(which='major', linestyle='-', linewidth=0.8) for _ in ax]
    ax[0].grid(which='minor', linestyle='-', linewidth=0.25)

    [_.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0)) for _ in ax]
    [_.set_xlabel('t') for _ in ax]
    fig.tight_layout()

    tri = Triangulation(arrays['nodes'][:,0], arrays['nodes'][:,1], arrays['connectivity'])
        
    #####################################################
    # CONTOUR PLOT C
    vmin = min(np.nanmin(d) for d in arrays['sol_c'][[0,nt]])
    vmax = max(np.nanmax(d) for d in arrays['sol_c'][[0,nt]])
    
    fig2, ax2 = plt.subplots(1,2, sharey=True)
    fig2.subplots_adjust(wspace=0.05)
    fig2.suptitle("C(x,y)")
    
    cf = ax2[0].tricontourf(tri, arrays['sol_c'][0] , vmin = vmin, vmax = vmax, levels = levels, cmap = cmap)
    cf = ax2[1].tricontourf(tri, arrays['sol_c'][nt], vmin = vmin, vmax = vmax, levels = levels, cmap = cmap)
    if plot_mesh:
        ax2[0].triplot(tri, linewidth = 0.5, color = 'k')
        ax2[1].triplot(tri, linewidth = 0.5, color = 'k')
        
    ax2[0].set_title(f"Starting Solution", fontsize = 10)
    ax2[1].set_title(f"Time = {arrays['t'][nt]:.2e}", fontsize = 10)
    for ax in ax2:
        ax.set_xlim(right = 0.99*ax.get_xlim()[1])
        ax.set_xlabel("x")
    ax2[0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    ax2[0].set_ylabel("y", rotation = 0, labelpad = 10)

    # for ax in ax2:
    #     # ax.spines['bottom'].set_zorder(0)
    #     [ax.spines[_].set_visible(False) for _ in ['right', 'top']]
    # # ax2[1].set_yticks([])
    # [ax2[1].spines[_].set_visible(False) for _ in ['left']]

    # Horizontal colorbar at bottom
    cbar = fig2.colorbar(
        cf,
        ax=ax2,
        orientation="horizontal",
        fraction=0.05,   # thickness of colorbar
        # pad=0.15         # distance from subplots
    )
    cbar.set_ticks(np.linspace(vmin, vmax, 5))  # fewer ticks → more spacing
    # fig2.tight_layout()
    
    
    # #####################################################
    # # CONTOUR PLOT W
    # fig3, ax3 = plt.subplots(1,2, sharey=True)
    # fig3.subplots_adjust(wspace=0.05)
    # fig3.suptitle("w(x,y)")
    # cf = ax3[0].tricontourf(tri, arrays['sol_w'][0], levels = levels, cmap = cmap)
    # fig3.colorbar(cf, ax=ax3[0], location='left')
    # cf = ax3[1].tricontourf(tri, arrays['sol_w'][nt], levels = levels, cmap = cmap)
    # fig3.colorbar(cf, ax=ax3[1], location='right')
    # if plot_mesh:
    #     ax3[0].triplot(tri, linewidth = 0.5, color = 'k')
    #     ax3[1].triplot(tri, linewidth = 0.5, color = 'k')

    # ax3[0].set_title(f"Starting Solution", fontsize = 10)
    # ax3[1].set_title(f"Time = {arrays['t'][nt]:.2e}", fontsize = 10)
    # for ax in ax3:
    #     ax.set_xlim(right = 0.99*ax.get_xlim()[1])
    #     ax.set_xlabel("x")
    # ax3[0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    # ax3[0].set_ylabel("y", rotation = 0, labelpad = 10)


def analyzer_time_complexity(prefix, dts, fp = Path.cwd() / "solution"):

    name_list = []
    for dt in dts:
        name_list.append(prefix + f"_dt{dt:.1e}.h5")

    ####################################################################
    # COMPARISON OF 1st ORDER STABILIZED SEMI IMPLICIT METHOD    
    complexity_data = []
    for i,name in enumerate(name_list):
        arrays, scalars = load_solution_hdf5(fp / name)

        # EXTRACT DATA
        x, t = arrays['x'], arrays['t']
        sol_c, sol_w = arrays['sol_c'], arrays['sol_w']
        nt = scalars['dt']  
        T = scalars['T']  
        
        # COMPLEXITY DATA
        if t[-1] == T:
            complexity_data.append([T/nt, sol_c[-1,:]])

    ##########################################
    # COMPLEXITY PLOT
    nt = [_[0] for _ in complexity_data[:-1]]
    error = [np.linalg.norm(_[1] - complexity_data[-1][1]) for _ in complexity_data[:-1]]
    fig3, ax3 = plt.subplots()
    ax3.loglog(nt, error, '-s')
    m,c = np.polyfit(np.log(nt), np.log(error), 1)
    def f(x): return x**(m)*np.exp(c)
    ax3.plot(nt, f(nt), '--r', label = r"$\log(e) = {:.2f}\log(nt) + {:.2f}$".format(m,c))
    
    ax3.grid(which='major', linestyle='-', linewidth=0.8)
    ax3.grid(which='minor', linestyle='-', linewidth=0.25)
    
    ax3.legend()

    ax3.set_xlabel("Time steps")
    ax3.set_ylabel(r"$||c(x, T; dt) - c(x, T; 1\times10^{-7})||_2$")
    ax3.set_title("Temporal Computational Complexity")
    fig3.tight_layout()

def analyzer_space_complexity(prefix, dt,fp = Path.cwd() / "solution"):

    pattern = prefix + f"*T1.0e-03_dt{dt:.1e}*"
    file_list = []
    for f in list(fp.rglob(pattern)):
        file_list.append(f)
    if len(file_list) == 0:
        raise RuntimeError(f"No file found with patter: '{pattern}'")
    ####################################################################
    # COMPARISON OF 1st ORDER STABILIZED SEMI IMPLICIT METHOD    
    complexity_data = []
    for i,name in enumerate(file_list):
        arrays, scalars = load_solution_hdf5(fp / name)

        # EXTRACT DATA
        sol_c, sol_w = arrays['sol_c'], arrays['sol_w']
        
        # COMPLEXITY DATA
        if arrays['t'][-1] == scalars['T']:
            complexity_data.append([scalars['Ne'], sol_c[-1][-1]])

    ##########################################
    # COMPLEXITY PLOT

    ne = np.array([_[0] for _ in complexity_data[:-1]])
    id = np.argsort(ne)
    ne_crit = 1/(2*np.sqrt(2)/9 *np.arctanh(0.95)*0.01)
    id2 = sorted([i for i, v in enumerate(ne) if v < ne_crit], key=lambda i: ne[i])
    
    error = np.array([abs(_[1] - complexity_data[-1][1]) for _ in complexity_data[:-1]])
    fig3, ax3 = plt.subplots()
    ax3.loglog(ne[id], error[id], '-s')
    m,c = np.polyfit(np.log(ne[id2]), np.log(error[id2]), 1)
    def f(x): return x**(m)*np.exp(c)
    ax3.plot(ne[id2], f(ne[id2]), '--r', label = r"$\log(e) = {:.2f}\log(nt) + {:.2f}$".format(m,c))
    
    ax3.axvline(ne_crit, color = 'k', label = r"$N_{e,crit} = \frac{9}{2\sqrt{2}\tanh^{-1}(0.95)}$")
        
    ax3.grid(which='major', linestyle='-', linewidth=0.8)
    ax3.grid(which='minor', linestyle='-', linewidth=0.25)
    
    ax3.legend()

    ax3.set_xlabel("Number of Elements")
    ax3.set_ylabel(r"$||c(x, T; dt) - c(x, T; 1\times10^{-7})||_2$")
    ax3.set_title("Spatial Computational Complexity")
    fig3.tight_layout()


if __name__ == '__main__':
    
    filename = "Cahn_Hilliard2D_solution_1ssi_Ne5824_T5.0e-03_dt1.0e-05.h5"


    # filename = 'circular_Cahn_Hilliard2D_solution_1ssi'
    # filename = "circular_Cahn_Hilliard2D_solution_1ssi_Ne18373_T1.0e-03_dt1.0e-05.h5"
    analyzer_contours(filename, nt = -1, plot_mesh=False)
    


    plt.show()
    