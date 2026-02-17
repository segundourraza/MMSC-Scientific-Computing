import h5py, re
from pathlib import Path
from typing import Dict, Tuple, List 
from collections import defaultdict, Counter

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

def group_by_ne_and_dt(files):
    """
    files: iterable of Path or strings (file names or pathlib.Path)
    returns: (ne_groups, dt_groups) where each is a dict: key -> list[Path]
    """
    # patterns: capture anything after Ne (or dt) until an underscore or end-of-string
    re_ne = re.compile(r'Ne([^_]+)')    # capture after 'Ne' until '_' (greedy non-underscore)
    re_dt = re.compile(r'dt(.*?)\.h5')    # capture after 'dt' until '.h5'

    groups: Dict[Tuple[int, str], List[Path]] = {}
    ne_counter = Counter()
    dt_counter = Counter()
    
    for p in files:
        p = Path(p)
        name = p.name

        m_ne = re_ne.search(name)
        m_dt = re_dt.search(name)

        if not m_ne:
            # skip files without Ne (change behavior if you want them included)
            continue

        # parse ne_key (try int conversion, otherwise leave as string)
        ne_key_raw = m_ne.group(1)
        try:
            ne_key = int(ne_key_raw)
        except Exception:
            # keep original string if it is not integer
            ne_key = ne_key_raw

        # parse dt_key (float if found), otherwise None
        dt_key = None
        if m_dt:
            dt_raw = m_dt.group(1)
            try:
                dt_key = float(dt_raw)   # supports scientific notation
            except Exception:
                dt_key = dt_raw  # fallback to raw string if float conversion fails

        pair = (ne_key, dt_key)
        groups[pair] = p
        ne_counter[ne_key] += 1
        dt_counter[dt_key] += 1

    def top_items(counter: Counter):
        if not counter:
            return []
        max_count = max(counter.values())
        return [k for k, c in counter.items() if c == max_count]
    ne_ch = top_items(ne_counter)
    dt_ch = top_items(dt_counter)
    if len(ne_ch) > 1:
        ne_ch = []
    
    if len(dt_ch) > 1:
        dt_ch = []

    return groups, ne_ch, dt_ch

def result_analyzer(prefix, period, fp = Path.cwd() / 'solution'):

    # FIND FILES
    pattern = prefix + f"_*T{period:.1e}*"
    file_list = []
    for f in list(fp.rglob(pattern)):
        file_list.append(f)
    if len(file_list) == 0:
        raise RuntimeError(f"No file found with pattern: '{pattern}'")

    grouped_files, ne_ch, dt_ch = group_by_ne_and_dt(file_list)
    print(dt_ch, ne_ch)
    ##########################################
    # TEMPORAL COMPLEXITY ANALYSIS
    complexity_data = []

    for i,name in enumerate(v for k,v in grouped_files.items() if k[0] in ne_ch):
        
        arrays, scalars = load_solution_hdf5(fp / name)

        # COMPLEXITY DATA
        if arrays['t'][-1] == scalars['T']:
            print(name)
            complexity_data.append([scalars['T']/scalars['dt'], arrays['sol_c'][-1,:]])
    
    if len(complexity_data) < 2:
        print('\nNot enough data to do a temporal complexity analysis')
    else:
        nt, complexity_data = zip(*sorted(complexity_data))
        nt = nt[:-1]
        error = [np.linalg.norm(_ - complexity_data[-1]) for _ in complexity_data[:-1]]
        
        # PLOTS
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


    ##########################################
    # COMPLEXITY PLOT

    ##########################################
    # SPATIAL COMPLEXITY ANALYSIS
    complexity_data = []
    for i,name in enumerate(v for k,v in grouped_files.items() if k[1] in dt_ch):
        arrays, scalars = load_solution_hdf5(fp / name)
        # COMPLEXITY DATA
        if arrays['t'][-1] == scalars['T']:
            complexity_data.append([scalars['Ne'], arrays['sol_c'][-1][-1]])
    if len(complexity_data) < 2:
        print('\nNot enough data to do a spatial complexity analysis')
    else:
        # # COMPLEXITY PLOT
        ne, complexity_data = zip(*sorted(complexity_data))
        ne = np.array(ne[:-1])
        error = np.array([abs(_ - complexity_data[-1]) for _ in complexity_data[:-1]])

        ne_crit = 1/(2*np.sqrt(2)/9 *np.arctanh(0.95)*0.01)
        # id2 = sorted([i for i, v in enumerate(ne) if v < ne_crit], key=lambda i: ne[i])
        id2 = range(len(ne))
        
        fig3, ax3 = plt.subplots()
        ax3.loglog(ne, error, '-s')
        m,c = np.polyfit(np.log(ne[id2]), np.log(error[id2]), 1)
        def f(x): return x**(m)*np.exp(c)
        ax3.plot(ne[id2], f(ne[id2]), '--r', label = r"$\log(e) = {:.2f}\log(Ne) + {:.2f}$".format(m,c))
        
        ax3.axvline(ne_crit, color = 'k', label = r"$N_{e,crit} = \frac{9}{2\sqrt{2}\tanh^{-1}(0.95)}$")
            
        ax3.grid(which='major', linestyle='-', linewidth=0.8)
        ax3.grid(which='minor', linestyle='-', linewidth=0.25)
        
        ax3.legend()

        ax3.set_xlabel("Number of Elements")
        ax3.set_ylabel(r"$||c(x, T; dt) - c(x, T; 1\times10^{-7})||_2$")
        ax3.set_title("Spatial Computational Complexity")
        fig3.tight_layout()


def result_visualizer(prefix, period, levels = 100, cmap = 'jet', fp = Path.cwd() / 'solution'):

    # FIND FILES
    pattern = prefix + f"_*T{period:.1e}*"
    file_list = []
    for f in list(fp.rglob(pattern)):
        file_list.append(f)
    if len(file_list) == 0:
        raise RuntimeError(f"No file found with pattern: '{pattern}'")

    grouped_files, ne_ch, dt_ch = group_by_ne_and_dt(file_list)
    





    ##############################################################################
    # COMPARISON OF DT    
    fcontour_data = []
    fig1, ax1 = plt.subplots(1,2)
    fig1.suptitle(f"Fix $N_e$={ne_ch[0]}, Varying $\\Delta t$")
    for i,name in enumerate(v for k,v in grouped_files.items() if k[0] in ne_ch):
        arrays, scalars = load_solution_hdf5(fp / name)

        # EXTRACT DATA
        x, t = arrays['x'], arrays['t']
        sol_c, sol_w = arrays['sol_c'], arrays['sol_w']
        mass, J = arrays['mass'], arrays['J']
        
        # FOR CONTOUR PLOTS
        X, Y = np.meshgrid(x, t)
        fcontour_data.append([X, Y, sol_c, sol_w, scalars['dt'], scalars['Ne']])
        
        ax1[0].semilogy(t[1:], abs(mass[1:]- mass[0]), label = f"$\\Delta$t = {scalars['dt']:.2e}")
        ax1[1].plot(t, J, label =  f"$\\Delta$t = {scalars['dt']:.2e}")

        
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

        
    #####################################################
    # CONTOUR PLOT C
    vmin = max(-1, min(np.nanmin(d[2]) for d in fcontour_data))
    vmax = min(1.0, max(np.nanmax(d[2]) for d in fcontour_data))

    fig2, ax2 = plt.subplots(1,len(fcontour_data),sharey=True)
    fig2.subplots_adjust(wspace=0.05)
    fig2.suptitle(f"$N_e$ = {ne_ch[0]}")
    # SORTING
    fcontour_data = sorted(fcontour_data, key = lambda p: p[-2], reverse=True)
    for i,(ax, data) in enumerate(zip(ax2, fcontour_data)):
        ax = ax2[i]
        X, Y, Z, _, dt, ne = data
        im = ax.contourf(X, Y, Z, levels = levels, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(f"$\\Delta$t = {dt:.2e}", fontsize = 10)
        ax.set_xlabel("x")
        ax.set_xlim(right = 0.99)
    ax2[0].set_ylabel("t", rotation = 0, labelpad = 10)
    ax2[0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # Horizontal colorbar at bottom
    cbar = fig2.colorbar(
        im,
        ax=ax2,
        orientation="horizontal",
        fraction=0.05,   # thickness of colorbar
        # pad=0.15         # distance from subplots
    )
    cbar.set_ticks(np.linspace(vmin, vmax, 5))  # fewer ticks → more spacing







    ##############################################################################
    # COMPARISON OF Ne
    
    
    fcontour_data = []
    fig1, ax1 = plt.subplots(1,2)
    fig1.suptitle(f"Fix $\\Delta t$={dt_ch[0]:.2e}, Varying Ne")
    for i,name in enumerate(v for k,v in grouped_files.items() if k[1] in dt_ch):
        arrays, scalars = load_solution_hdf5(fp / name)

        # EXTRACT DATA
        x, t = arrays['x'], arrays['t']
        sol_c, sol_w = arrays['sol_c'], arrays['sol_w']
        mass, J = arrays['mass'], arrays['J']
        
        # FOR CONTOUR PLOTS
        X, Y = np.meshgrid(x, t)
        fcontour_data.append([X, Y, sol_c, sol_w, scalars['dt'], scalars['Ne']])
        
        ax1[0].semilogy(t[1:], abs(mass[1:]- mass[0]), label = f"$N_e$ = {scalars['Ne']}")
        ax1[1].plot(t, J, label =  f"$N_e$ = {scalars['Ne']}")

        
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

        
    #####################################################
    # CONTOUR PLOT C
    vmin = max(-1, min(np.nanmin(d[2]) for d in fcontour_data))
    vmax = min(1.0, max(np.nanmax(d[2]) for d in fcontour_data))

    fig2, ax2 = plt.subplots(1,len(fcontour_data),sharey=True)
    fig2.subplots_adjust(wspace=0.05)
    fig2.suptitle(f"$\\Delta$ t = {dt_ch[0]:.2e}")
    # SORTING
    fcontour_data = sorted(fcontour_data, key = lambda p: p[-1])
    for i,(ax, data) in enumerate(zip(ax2, fcontour_data)):
        ax = ax2[i]
        X, Y, Z, _, dt, ne = data
        im = ax.contourf(X, Y, Z, levels = levels, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(f"$N_e$ = {ne}", fontsize = 10)
        ax.set_xlabel("x")
        ax.set_xlim(right = 0.99)
    ax2[0].set_ylabel("t", rotation = 0, labelpad = 10)
    ax2[0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # Horizontal colorbar at bottom
    cbar = fig2.colorbar(
        im,
        ax=ax2,
        orientation="horizontal",
        fraction=0.05,   # thickness of colorbar
        # pad=0.15         # distance from subplots
    )
    cbar.set_ticks(np.linspace(vmin, vmax, 5))  # fewer ticks → more spacing



if __name__ == '__main__':
    
    # # SCHEME B
    # prefix = "Cahn_Hilliard_solution_b_CG1_Ne100_T1.0e-03"
    # dts = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    
    # # 1SI
    # prefix = "Cahn_Hilliard_solution_1si_CG1_Ne100_T1.0e-03"
    # dts = [1e-5, 1e-6, 5e-7, 1e-7, 5e-8]
    # # dts = [1e-5]
    # # dts = [1e-6, 5e-7, 1e-7, 5e-8]
    
    
    # # 1SSI
    # prefix = "Cahn_Hilliard_solution_1ssi_CG1_Ne100_T1.0e-03"
    # dts = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    # analyzer_contours_with_dt(prefix, dts)
    # analyzer_time_complexity(prefix, dts)
    
    # 2SSI
    # prefix = "Cahn_Hilliard_solution_2ssi_CG2_Ne100_T1.0e-03"
    # prefix = "Cahn_Hilliard_solution_2ssi_CG1_Ne100_T1.0e-03"
    # dts = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    # analyzer_contours_with_dt(prefix, dts)
    # analyzer_time_complexity(prefix, dts)
    
    # prefix = "Cahn_Hilliard_solution_1ssi_CG1"
    # dt = 1e-5
    # analyzer_contours_with_Ne(prefix, dt)
    # # analyzer_space_complexity(prefix, dt)
    
    
    # prefix = "Cahn_Hilliard_solution_1ssi_CG2"
    # dt = 1e-5
    # analyzer_contours_with_Ne(prefix, dt)
    # analyzer_space_complexity(prefix, dt)
    
    
    prefix = "Cahn_Hilliard_solution_b_CG1"
    prefix = "Cahn_Hilliard_solution_1si_CG1"
    # prefix = "Cahn_Hilliard_solution_1ssi_CG1"
    # prefix = "Cahn_Hilliard_solution_1ssi_CG2"
    # prefix = "Cahn_Hilliard_solution_2ssi_CG1"
    # prefix = "Cahn_Hilliard_solution_2ssi_CG2"
    period = 1e-3
    result_analyzer(prefix, period)
    result_visualizer(prefix, period)

    plt.show()
    