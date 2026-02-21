import h5py
from datetime import timezone, datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.sparse import csc_matrix, bmat
import scipy.sparse.linalg as linalg
from matplotlib.animation import FuncAnimation, PillowWriter

from ._elements import LinearTriangularElement, LinearRectElement, QuadraticRectElement
from ._config import _progress_range, STABILIZATION_CONSTANT, tqdm
from ._mesh import generate_circular_domain, generate_rectangular_domain, generate_rect_mesh

TIME_INTEGRATOR_STR2INT_MAP = {
                                'implicit': 1,
                                'b': 3, 
                                '1si': 4,
                                '1ssi' : 5,
                                '2ssi' : 6,
                                }

TIME_INTEGRATOR_INT2STR_MAP = {
                               1: 'implicit',
                               3: 'b',
                               4: '1si',
                               5: '1ssi',
                               6: '2ssi',
                               }

class CahnHilliardSolver2D:

    def __init__(self, epsilon:float, nodes: np.ndarray, connectivity: np.ndarray):
        
        self.epsilon = epsilon

        # Nodes and connectivity
        self.__nodes = nodes
        self.__connectivity = connectivity

        self.__n = len(self.__connectivity[0])
        if self.__n == 3:
            self.element = LinearTriangularElement()
            self.__tri = Triangulation(self.__nodes[:,0], self.__nodes[:,1], self.__connectivity)
        elif self.__n == 4:
            self.element = LinearRectElement()
            self.__tri = None
        elif self.__n == 9:
            self.element = QuadraticRectElement()
        else:
            raise ValueError(f"No compatible element for a {self.__n} point element.")

        # Number of nodes and elements
        self.__N = len(self.__nodes)
        self.__Ne = len(self.__connectivity)

        # Preprocessing
        self.__preprocessing()

    def __preprocessing(self,):

        # Compute Jacobian
        self.__detJ = np.zeros((self.__Ne,))
        self.__InvJ = [0]*self.__Ne
        for e,con in enumerate(self.__connectivity):
            self.__detJ[e], self.__InvJ[e] = self.element.compute_ele_properties(self.__nodes[con])
            

        # Evaluate 'Mass' and 'Stiffness' matrix. These DO NOT change with time or value of C
        self.M = self.__assemble_M()
        self.K = self.__assemble_K()

    
    def solve(self, u0, T:float, dt:float, time_integrator:str|int = 'explicit', nonlinear_solver_options:dict = {}, 
              terminate_solver = True):
        
        self.__T = T
        self.__dt = dt
        self.__t = np.arange(0, T+dt, dt)
        self.__nt = len(self.__t)

        if isinstance(time_integrator, str):
            time_integrator = TIME_INTEGRATOR_STR2INT_MAP[time_integrator.lower()]
        self.__solver_name = TIME_INTEGRATOR_INT2STR_MAP[time_integrator]
        match time_integrator:
            case 3:
                _time_stepper = self._semi_implicit_B
            case 5:
                _time_stepper = self._1SSI_scheme
            case 6:
                _time_stepper = self._2SSI_scheme
            case _:
                raise ValueError
            

        ###########
        self.__nonlinear_solver_parameters = {k:v for k,v in nonlinear_solver_options.items() if v is not None}
        


        # CONSTRUCT SOLUTION VECTOR U = [C , W]
        self.__u = np.zeros((self.__nt, 2*self.__N), dtype=float)
        if callable(u0):
            self.__u[0][:self.__N] = u0(self.__nodes[:,0], self.__nodes[:,1])
        elif len(u0) == self.__N:
            self.__u[0][:self.__N] = u0

        # Computation of W[0] via the variational problem
        N0 = self.__assemble_N(self.__u[0,:self.__N])
        b = self.epsilon*(self.K@self.__u[0,:self.__N]) + 1/self.epsilon * N0
        self.__u[0,self.__N:] = linalg.spsolve(self.M, b)
        
        # CONSERVED QUANTITIES
        self.__mass = np.zeros((self.__nt,), dtype=float)
        self.__energy = np.zeros((self.__nt,), dtype=float)
        self.__mass[0] = self.__compute_mass(self.__u[0])
        self.__energy[0] = self.__compute_energy(self.__u[0])

        
        ################################################
        # TIME STEPPING
        # return
        print()
        self.__termination_flag = _time_stepper(terminate_solver)
        tqdm.write("Simulation ended.\n")
        
    
    
    ########################################################################
    # SEMI-IMPLICIT B TIME STEPPING
    def _semi_implicit_B(self,terminate_solver):
        """Semi-Implicit methods Case B"""

        b = np.zeros((self.__N*2,))

        # Used for Semi-Implicit methods B
        ck = -self.epsilon*self.K - (2/self.epsilon)*self.M
        # Compute LHS: This is done once for stencil B
        A = bmat([[ck,     self.M],
                  [self.M, self.__dt*self.K]], 
                  format= 'csc')
        # A = bmat([[self.M/self.__dt, self.K],
        #           [ck,               self.M]], 
        #           format='csc')
        
        lu = linalg.splu(A)
        for it in _progress_range(range(1, self.__nt), desc = "Simulation running"):
            # Update RHS
            self.__update_rhs_B(b, self.__u[it-1,:self.__N])

            # SOLVE
            self.__u[it] = lu.solve(b)

            flag = self.__checks(it,terminate_solver)

            if flag != 0:
                return 1
        return 0 
        
    def __update_rhs_B(self, b, evaluation_C):
        """Assemble non-linear vector for semi implicit B"""
        b[self.__N:] = (self.M@evaluation_C)
        b[:self.__N] = 0
        for e,con in enumerate(self.__connectivity):
            self.element.b2_b(b, con ,self.__detJ[e], evaluation_C[con])
        b[:self.__N] *= 1/self.epsilon

    
    ########################################################################
    # 1ST-ORDER STABILIZED SEMI-IMPLICIT SCHEME (1SSI)

    def _1SSI_scheme(self,terminate_solver):
        """
        1st-order Stabilized Semi-Implicit Scheme (1SSI)
        
        Ref: NUMERICAL APPROXIMATIONS OF ALLEN-CAHN AND CAHN-HILLIARD EQUATIONS - Jie Shen
        """
        # Compute LHS: This is done once for stencil B
        A11 = -self.epsilon*self.K - STABILIZATION_CONSTANT/self.epsilon*self.M
        A = bmat([[A11 ,    self.M],
                  [self.M,  self.__dt*self.K]], format='csc')
        b = np.zeros((self.__N*2,))
        
        # Pre-compute LU factorisation
        lu = linalg.splu(A)
        for it in _progress_range(range(1,self.__nt),desc = "Simulation running"):
            # Update RHS
            self.__update_rhs_1SSI(b, self.__u[it-1,:self.__N])
            
            # SOLVE
            self.__u[it] = lu.solve(b)

            flag = self.__checks(it,terminate_solver)

            if flag != 0:
                return self.__u[:it-1,:]
        
    def __update_rhs_1SSI(self, b, evaluation_C):
        """Assemble non-linear vector for 1st order stabilized semi-implicit scheme"""
        temp = (self.M@evaluation_C)
        b[self.__N:] = temp
        b[:self.__N] = -temp - STABILIZATION_CONSTANT*temp
        for e,con in enumerate(self.__connectivity):
            self.element._c3(b, con, self.__detJ[e], evaluation_C[con])
        b[:self.__N] *= 1/self.epsilon


    ########################################################################
    # 2ND ORDER STABILIZED SEMI-IMPLICIT SCHEME (2SSI)

    def _2SSI_scheme(self, terminate_solver):
        """
        2nd-order Stabilized Semi-Implicit Scheme (2SSI)
        
        Ref: NUMERICAL APPROXIMATIONS OF ALLEN-CAHN AND CAHN-HILLIARD EQUATIONS - Jie Shen
        """
    
        # USE A TIME STEP OF 1SSI
        # Since the scheme is second order, we require two previous computations to propagate solution
        # Compute LHS: This is done once for stencil B
        dt1 = 1e-6
        A11 = -self.epsilon*self.K - STABILIZATION_CONSTANT/self.epsilon*self.M
        A = bmat([[A11 ,    self.M],
                  [self.M,  dt1*self.K]], format='csc')
        b = np.zeros((self.__N*2,))
        
        # Pre-compute LU factorisation
        lu = linalg.splu(A)

        # Update RHS
        self.__update_rhs_1SSI(b, self.__u[0,:self.__N])
        # Solve
        u2 = lu.solve(b)
        

        # START USING 2SSI  
        A11 = -self.epsilon*self.K - STABILIZATION_CONSTANT/self.epsilon*self.M
        A = bmat([[A11 ,    self.M],
                  [self.M,  2/3*self.__dt*self.K]], format='csc')
        lu = linalg.splu(A)

        self.__update_rhs_2SSI(b, self.__u[0,:self.__N], u2[:self.__N])
        
        # Solve
        self.__u[1] = lu.solve(b)
        
        # Run checks
        flag = self.__checks(1, terminate_solver=terminate_solver)

        
        
        for it in _progress_range(range(2,self.__nt), desc = "Simulation running"):
            # Update RHS
            self.__update_rhs_2SSI(b, self.__u[it-2,:self.__N], self.__u[it-1,:self.__N])
            
            # SOLVE
            self.__u[it] = lu.solve(b)

            flag = self.__checks(it,terminate_solver)

            if flag != 0:
                return self.__u[:it-1,:]
        
    def __update_rhs_2SSI(self, b, evaluation_C1, evaluation_C2):
        """Assemble non-linear vector for 1st order stabilized semi-implicit scheme"""
        phi1 = np.zeros((self.__N,))
        phi2 = np.zeros((self.__N,))
        for e,con in enumerate(self.__connectivity):
            self.element._c3(phi1, con,self.__detJ[e], evaluation_C1[con])
            self.element._c3(phi2, con,self.__detJ[e], evaluation_C2[con])
        b[:self.__N] = -2*(STABILIZATION_CONSTANT + 1)*self.M@evaluation_C2 + 2*phi2 +\
                          (STABILIZATION_CONSTANT + 1)*self.M@evaluation_C1 - phi1
        b[:self.__N] *= 1/self.epsilon
        b[self.__N:] = self.M@(1/3*(4*evaluation_C2 - evaluation_C1))

    ####################################################################
    # ASSEMBLE GLOBAL LINEAR SYSTEMS
    def __assemble_M(self)->csc_matrix:
        """Assemble Mass Matrix"""
        M = np.zeros((self.__N, self.__N), dtype= float)
        # Loop over elements
        for e,con in enumerate(self.__connectivity):
            self.element.Me(M, con, self.__detJ[e])
        return csc_matrix(M)
    
    def __assemble_K(self)->csc_matrix:
        """Assemble Stiffness Matrix"""
        K = np.zeros((self.__N, self.__N), dtype= float)
        for e,con in enumerate(self.__connectivity):
            self.element.Ke(K, con, self.__detJ[e], self.__InvJ[e])
        return csc_matrix(K)    
    
    def __assemble_N(self, evaluation_C):
        """Assemble non-linear mass matrix"""
        N = np.zeros((self.__N,), dtype=float)
        for e,con in enumerate(self.__connectivity):
            # Classic overlapping block assembly
            self.element.Ne(N, con, self.__detJ[e], evaluation_C[con])
        return N
    
    #####################################################################
    # CONSTRUCTORS
    @classmethod
    def rectangular_domain_tri(cls, epsilon, height, width, mesh_size = 0.08):
        """
        Generate a 2D triangular mesh of a rectangle height x width.
        """
        nodes, connectivity = generate_rectangular_domain(height=height, width=width, mesh_size=mesh_size)
        return cls(epsilon=epsilon, nodes=nodes, connectivity=connectivity)
    
    @classmethod
    def rectangular_domain_rect(cls, epsilon, height, width, nx, ny, order):
        """
        Generate a 2D triangular mesh of a rectangle height x width.
        """
        nodes, connectivity = generate_rect_mesh(nx, ny, width, height, order=order)
        return cls(epsilon=epsilon, nodes=nodes, connectivity=connectivity)
    
    @classmethod
    def generate_circular_mesh(cls, epsilon, r=1.0, mesh_size=0.1):
        """
        Generate a 2D triangular mesh of a disk of radius r.        
        """
        nodes, connectivity = generate_circular_domain(radius=r, mesh_size=mesh_size)
        return cls(epsilon=epsilon, nodes=nodes, connectivity=connectivity)

    #####################################################################
    # AUXILIARY FUNCTIONS
    def plot_mesh(self, ax = None, linewidth = 0.6, color = 'k', plot_nodes = False, node_color = 'k', node_size = 6, **kwargs):
        if ax is None:
            ax = plt.gca()  
        if plot_nodes:
            ax.plot(self.__nodes[:,0], self.__nodes[:,1], '.', color = node_color, ms = node_size)
        
        if self.__tri:
            ax.triplot(self.__tri, linewidth = linewidth, color = color)
        else:
            if self.element.n == 9:
                end = self.__n-1
            else:
                end = self.__n

            for e, con in enumerate(self.__connectivity):
                temp = np.vstack([self.__nodes[con[:end]],self.__nodes[con[0]]])
                plt.plot(*temp, '-', color = color, linewidth= linewidth)


    def plot_solution(self, z, ax = None, cmap = 'jet', levels = 100, plot_mesh = False, **kwargs):
        if ax is None:
            ax = plt.gca()  
            
        vmin = kwargs.get('vmin', min(np.nanmin(z), -1.0))
        vmax = kwargs.get('vmax', max(np.nanmax(z), 1.0))

        levels = np.linspace(vmin, vmax, levels)
        tcf = ax.tricontourf(self.__tri, z, levels, cmap = cmap)
        if plot_mesh:
            self.plot_mesh(ax=ax, **kwargs)
        return tcf, levels

    def animate_solution(self, vector = 'c', fps = 10, cmap = 'jet', levels = 100, prepend = None, directory = None):

        if vector == 'c':
            v = self.sol_c
        elif vector == 'w':
            v = self.sol_w
        else:
            RuntimeError()
        
        if prepend is None:
            filename = self.simulation_name + "_gif_" + vector
        else:
            filename = prepend + "_" + self.simulation_name + "_gif_" + vector 

        # Ensure extension
        if not filename.endswith(".gif"):
            filename += ".gif"
        
        # Determine directory
        if directory is None:
            directory = Path.cwd() / "gifs"
        else:
            directory = Path(directory)

        # Create directory if it does not exist
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / filename

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("tricontourf GIF example")

        vmin = np.nanmin(v)
        vmax = np.nanmax(v)
        tcf = ax.tricontourf(self.__tri, v[0], levels, cmap = cmap, vmin= vmin, vmax = vmax)
        ax.set_title(f"Tme step: 0")
        cbar = fig.colorbar(tcf, ax=ax)
        
        def update(i):
            ax.clear()
            tcf = ax.tricontourf(self.__tri, v[i], levels=levels, cmap = cmap, vmin= vmin, vmax = vmax)
            # cbar.update_normal(tcf)
            ax.set_title(f"Time step: {i}")
            return tcf
        
        anim = FuncAnimation(fig, update, frames=self.__nt, interval=100, blit=False)

        # Save as GIF
        writer = PillowWriter(fps=fps)   # frames per second
        # pbar = tqdm(total= self.__nt, leave= LEAVE_TQDM_BAR, desc= "Animating solution:")
        pbar = _progress_range(range(self.__nt), f"Animating '{vector}' solution")
        
        def progress(i, n):
            pbar.update(1)
        
        anim.save(filepath,writer=writer,dpi=150,progress_callback=progress)
        pbar.close()
        tqdm.write(f"File saved successfully: {filepath}\n")
        plt.close(fig)
        
    
    def save(self, prepend = None, directory = None, append_time = False):
        if prepend is None:
            filename = self.simulation_name
        else:
            filename = prepend + "_" + self.simulation_name

        if append_time:
            filename += "_" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")

        # Ensure extension
        if not filename.endswith(".h5"):
            filename += ".h5"
        
        # Determine directory
        if directory is None:
            directory = Path.cwd() / "solution2D"
        else:
            directory = Path(directory)

        # Create directory if it does not exist
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / filename

        with h5py.File(filepath, "w") as f:
            sol_grp = f.create_group("solution")
            
            # -----------------
            # Save arrays
            # -----------------
            arr_grp = sol_grp.create_group("arrays")
            for name, array in zip(['sol_c', 'sol_w', 't', 'mass', 'energy', 'nodes', 'connectivity'], [self.sol_c, self.sol_w, self.t, self.__mass, self.__energy, self.__nodes, self.__connectivity]):
                arr_grp.create_dataset(
                    name,
                    data=array,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True
            )


            # -----------------
            # Save scalars
            # -----------------
            scal_grp = sol_grp.create_group("scalars")
            for name, value in zip(['Ne', 'N', 'dt', 'T'], [self.__Ne, self.__N, self.__dt, self.__T]):
                scal_grp.create_dataset(name, data=value)

            # -----------------
            # Save metadata
            # -----------------
            sol_grp.attrs["saved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")
        
        print("Simulation successfully saved as : {}\n".format(filepath))

    #####################################################################
    # HELPER FUNCTIONS
    
    def __compute_mass(self, u):
        M = 0
        for e,con in enumerate(self.__connectivity):
            M += self.element.compute_mass(self.__detJ[e],u[con])
        return M
    
    def __compute_energy(self, u):
        E = 0
        for e, con in enumerate(self.__connectivity):
            E += self.element.compute_energy(self.__detJ[e], self.__InvJ[e], u[con], self.epsilon)
        return E
    
    def __checks(self, it, terminate_solver = True):
        # Compute Conserved quantities
        self.__mass[it] = self.__compute_mass(self.__u[it])
        self.__energy[it] = self.__compute_energy(self.__u[it])

        if terminate_solver:
            if not np.isclose(self.__mass[it], self.__mass[0]):
                tqdm.write(f"\nERROR IN ITERATION: {it:4d}")
                tqdm.write("MASS IS NOT BEING CONSERVED!!!!!!!!!!!!!!!!!!")
                tqdm.write(f"{self.__mass[it]:.6e}, {self.__mass[it]:.6e}")
                self.__t = self.__t[:it+1]
                self.__mass = self.__mass[:it+1]
                self.__energy = self.__energy[:it+1]
                self.__u = self.__u[:it+1,:]
                return 1
            elif (self.__energy[it]-self.__energy[it-1])/self.__energy[it-1] > 0.01:
                tqdm.write(f"\nERROR IN ITERATION: {it:4d}")
                tqdm.write("J INCREASING!!!!!!!!!!!!!!!!!!")
                tqdm.write(f"{self.__energy[it-1] - self.__energy[it]:.6e}")
                self.__t = self.__t[:it+1]
                self.__mass = self.__mass[:it+1]
                self.__energy = self.__energy[:it+1]
                self.__u = self.__u[:it+1,:]
                return 2

        return 0
    
    
    ######################################################
    # PROPERTIES

    @property
    def dt(self):
        """time step"""
        return self.__dt

    @property
    def nt(self):
        """Number of time steps"""
        return self.__nt
    
    @property
    def N(self):
        """Number of nodes"""
        return self.__N
    
    @property
    def Ne(self):
        """Number of elements"""
        return self.__Ne
        
    @property
    def t(self):
        """time array"""
        return self.__t
    
    @property
    def sol_c(self):
        """solution of c"""
        return self.__u[:,:self.__N]
    
    
    @property
    def sol_w(self):
        """solution of w"""
        return self.__u[:,self.__N:]
    
    @property
    def sol_u(self):
        """solution of w"""
        return self.__u
    
    @property
    def mass(self):
        """Mass"""
        return self.__mass
    
    
    @property
    def E(self):
        """Energy"""
        return self.__energy
    
    @property
    def tri(self):
        """Triangulations object"""
        return self.__tri

    @property
    def nodes(self):
        """Nodes"""
        return self.__nodes
        
    @property
    def connectivity(self):
        """connectivity"""
        return self.__connectivity
    
    @property
    def simulation_name(self):
        return f"Cahn_Hilliard2D_solution_{self.__solver_name}_Ne{self.Ne}_T{self.__T:.1e}_dt{self.__dt:.1e}"