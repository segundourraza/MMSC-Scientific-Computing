import h5py
from pathlib import Path

import matplotlib.pyplot as plt
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

    
def analyzer(prefix, dts, levels = 100, cmap = 'jet', fp = Path.cwd() / "solution"):

    name_list = []
    for dt in dts:
        name_list.append(prefix + f"_dt{dt:.1e}.h5")

    ####################################################################
    # COMPARISON OF 1st ORDER STABILIZED SEMI IMPLICIT METHOD    
    fcontour_data = []
    complexity_data = []
    fig1, ax1 = plt.subplots(1,2)
    for i,name in enumerate(name_list):
        arrays, scalars = load_solution_hdf5(fp / name)

        # EXTRACT DATA
        x, t = arrays['x'], arrays['t']
        sol_c, sol_w = arrays['sol_c'], arrays['sol_w']
        mass, J = arrays['mass'], arrays['J']
        nt = scalars['dt']  
        T = scalars['T']  
        
        # FOR CONTOUR PLOTS
        X, Y = np.meshgrid(x, t)
        fcontour_data.append([X, Y, sol_c])
        
        # COMPLEXITY DATA
        if t[-1] == T:
            complexity_data.append([T/nt, sol_c[-1,:]])

        ax1[0].semilogy(t[1:], abs(mass[1:]- mass[0]), label = f"$\\Delta$ t = {nt:.1e}")
        ax1[1].plot(t, J, label = f"$\\Delta$ t = {nt:.1e}")

        
    ax1[1].legend(fontsize = 12)
    ax1[0].set_title("Error in Mass")
    ax1[0].set_ylabel('$|\\mathcal{M}(c^n) - \\mathcal{M}(c^0)|$')
    
    ax1[1].set_title("Ginzburg-Landau Energy")
    ax1[1].set_ylabel('$\\mathcal{E}(c)$')

    [_.grid(which='major', linestyle='-', linewidth=0.8) for _ in ax1]
    ax1[0].grid(which='minor', linestyle='-', linewidth=0.25)

    [_.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0)) for _ in ax1]
    [_.set_xlabel('t') for _ in ax1]
    fig1.tight_layout()

        
    
    vmin = max(-1, min(np.nanmin(d) for d in fcontour_data))
    vmax = min(1.0, max(np.nanmax(d) for d in fcontour_data))

    fig2, ax2 = plt.subplots(1,len(dts),sharey=True)
    fig2.subplots_adjust(wspace=0.05)

    for i, (ax, data)in enumerate(zip(ax2, fcontour_data)):
        title = f"$\\Delta$t = {dts[i]:.1e}"
        X, Y, Z = data
        im = ax.contourf(X, Y, Z, levels = levels, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_xlim(right = 0.99)
    ax2[0].set_ylabel("t", rotation = 0, labelpad = 10)
    ax2[0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # Horizontal colorbar at bottom
    cbar = fig1.colorbar(
        im,
        ax=ax2,
        orientation="horizontal",
        fraction=0.05,   # thickness of colorbar
        # pad=0.15         # distance from subplots
    )
    cbar.set_ticks(np.linspace(vmin, vmax, 5))  # fewer ticks → more spacing


    ##########################################
    # COMPLEXITY PLOT
    nt = [_[0] for _ in complexity_data[:-1]]
    error = [np.linalg.norm(_[1] - complexity_data[-1][1]) for _ in complexity_data[:-1]]
    print(error)
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
    ax3.set_title("Computational Complexity")
    fig3.tight_layout()



if __name__ == '__main__':
    
    # 1SI
    prefix = "Cahn_Hilliard_solution_1s1_CG1_Ne100_T1.0e-03"
    dts = [1e-5, 1e-6, 5e-7, 1e-7, 5e-8]
    # dts = [1e-5]
    # dts = [1e-6, 5e-7, 1e-7, 5e-8]
    
    
    # 1SSI
    prefix = "Cahn_Hilliard_solution_1ss1_CG1_Ne100_T1.0e-03"
    dts = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    dts = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    
    analyzer(prefix, dts)
    

    plt.show()
    