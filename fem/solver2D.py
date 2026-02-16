import pygmsh, os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.sparse import csc_matrix, bmat
import scipy.sparse.linalg as linalg
from matplotlib.animation import FuncAnimation, PillowWriter

from ._elements import LinearTriangularElement
from ._config import _progress_range, LEAVE_TQDM_BAR, STABILIZATION_CONSTANT

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
        self.__tri = Triangulation(self.__nodes[:,0], self.__nodes[:,1], self.__connectivity)
        # Number of nodes and elements
        self.__N = len(self.__nodes)
        self.__Ne = len(self.__connectivity)
        
        # Check what element corresponds to the mesh
        if len(self.__connectivity[0]) == 3:
            self.element = LinearTriangularElement()
        else:
            raise ValueError(f"No compatible element for a {len(connectivity)} point element.")

        # Preprocessing
        self.__preprocessing()

    def __preprocessing(self,):

        # Compute Jacobian
        self.__J = [0]*self.__Ne
        self.__A = np.empty((self.__Ne,))
        self.__detJ = np.empty((self.__Ne,))
        self.__InvJ = [0]*self.__Ne
        for e,con in enumerate(self.__connectivity):
            x1, x2, x3 = self.__nodes[con,0]
            y1, y2, y3 = self.__nodes[con,1]

            dx31 = x3 - x1
            dx21 = x2 - x1
            dy21 = y2 - y1
            dy31 = y3 - y1

            self.__J[e] = np.array([[dx21, dx31],
                                    [dy21, dy31]])  # Jacobian            
            self.__detJ[e] = dy31*dx21 - dx31*dy21  # Det of jacobian
            self.__A[e] = 0.5 * self.__detJ[e]      # Area of cell
            self.__InvJ[e] = (1/self.__detJ[e])*np.array([[dy31, -dx31],
                                                          [-dy21, dx21]]) # Inverse of jacobian
            

        # Evaluate 'Mass' and 'Stiffness' matrix. These DO NOT change with time or value of C
        self.M = self.__assemble_M()
        self.K = self.__assemble_K()

    
    def solve(self, u0, T:float, dt:float, time_integrator:str|int = 'explicit', nonlinear_solver_options:dict = {}):
        
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
        self.__mass = np.empty((self.__nt,), dtype=float)
        self.__energy = np.empty((self.__nt,), dtype=float)
        self.__mass[0] = self.__compute_mass(self.__u[0])
        self.__energy[0] = self.__compute_energy(self.__u[0])

        
        ################################################
        # TIME STEPPING
        # return
        _time_stepper()
        tqdm.write("Simulation ended.\n")
        
    
    
    ########################################################################
    # SEMI-IMPLICIT B TIME STEPPING
    def _semi_implicit_B(self):
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

            flag = self.__checks(it)

            if flag != 0:
                return self.__u[:it-1,:]
        
    def __update_rhs_B(self, b, evaluation_C):
        """Assemble non-linear vector for semi implicit B"""
        b[self.__N:] = (self.M@evaluation_C)
        b[:self.__N] = 0
        for e,con in enumerate(self.__connectivity):
            self.element.b2_b(b, con ,self.__detJ[e], evaluation_C[con])
        b[:self.__N] *= 1/self.epsilon

    
    ########################################################################
    # 1st-order Stabilized Semi-Implicit Scheme (1SSI)

    def _1SSI_scheme(self):
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
            self.__update_rhs_1SSI(b, self.__u[it-1,:self.__N], STABILIZATION_CONSTANT)
            
            # SOLVE
            self.__u[it] = lu.solve(b)

            flag = self.__checks(it)

            if flag != 0:
                return self.__u[:it-1,:]
        
    def __update_rhs_1SSI(self, b, evaluation_C, S):
        """Assemble non-linear vector for 1st order stabilized semi-implicit scheme"""
        temp = (self.M@evaluation_C)
        b[self.__N:] = temp
        b[:self.__N] = -temp - S*temp
        for e,con in enumerate(self.__connectivity):
            self.element._c3(b, con, self.__detJ[e], evaluation_C[con])
        b[:self.__N] *= 1/self.epsilon


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
            self.element.Ne(N[con], self.__detJ[e], evaluation_C[con])
        return N
    
    #####################################################################
    # CONSTRUCTORS
    @classmethod
    def rectangular_domain(cls, epsilon, height, width, mesh_size = 0.08):
        
        poly_pts = [
            [0.0,   0.0,    0.0],
            [width, 0.0,    0.0],
            [width, height, 0.0],
            [0.0,   height, 0.0],
        ]

        with pygmsh.geo.Geometry() as geom:
            poly = geom.add_polygon(poly_pts, mesh_size=mesh_size)
            mesh = geom.generate_mesh()
        
        nodes = mesh.points 
        connectivity = None
        for cell_block in mesh.cells:
            if cell_block.type == "triangle":
                connectivity = cell_block.data
                break
        
        return cls(epsilon=epsilon, nodes=nodes, connectivity=connectivity)
    

    #####################################################################
    # AUXILIARY FUNCTIONS
    def plot_mesh(self, ax = None, linewidth = 0.6, color = 'k', plot_nodes = False, node_color = 'k', node_size = 6, **kwargs):
        if ax is None:
            ax = plt.gca()  
        if plot_nodes:
            ax.plot(self.__nodes[:,0], self.__nodes[:,1], '.', color = node_color, ms = node_size)
        
        ax.triplot(self.__tri, linewidth = linewidth, color = color)

    def plot_solution(self, z, ax = None, cmap = 'jet', levels = 100, plot_mesh = False, **kwargs):
        if ax is None:
            ax = plt.gca()  
            
        vmin = kwargs.get('vmin', np.floor(np.nanmin(z)))
        vmax = kwargs.get('vmax', np.ceil(np.nanmax(z)))
        
        levels = np.linspace(vmin, vmax, levels)
        tcf = ax.tricontourf(self.__tri, z, levels, cmap = cmap)
        if plot_mesh:
            self.plot_mesh(ax=ax, **kwargs)
        return tcf, levels

    def animate_solution(self, fps = 10, cmap = 'jet', levels = 100,  out_path="gifs/tricontourf_animation.gif"):
        
        levels = np.linspace(-1, 1, levels)

        # figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("tricontourf GIF example")

        tcf = ax.tricontourf(self.__tri, self.sol_c[0], levels, cmap = cmap)
        ax.set_title(f"Tme step: 0")
        cbar = fig.colorbar(tcf, ax=ax)
        
        def update(i):
            ax.clear()
            tcf = ax.tricontourf(self.__tri, self.sol_c[i], levels=levels, cmap = cmap)
            ax.set_title(f"Tme step: {i}")
            return tcf

        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            anim = FuncAnimation(fig, update, frames=self.__nt, interval=100, blit=False)

        # Save as GIF
        writer = PillowWriter(fps=fps)   # frames per second
        # pbar = tqdm(total= self.__nt, leave= LEAVE_TQDM_BAR, desc= "Animating solution:")
        pbar = _progress_range(range(self.__nt), "Animating solution")
        
        def progress(i, n):
            pbar.update(1)
        try:
            anim.save(out_path,writer=writer,dpi=150,progress_callback=progress)
        finally:
            pbar.close()
        tqdm.write(f"File saved successfully: {out_path}\n")
        plt.close(fig)
        
    #####################################################################
    # HELPER FUNCTIONS
    
    def __compute_mass(self, u):
        M = 0
        for e,con in enumerate(self.__connectivity):
            M += self.element.compute_mass_e(self.__A[e], u[con])
        return M
    
    def __compute_energy(self, u):
        return 1
    
    def __checks(self, it):
        # Compute Conserved quantities
        self.__mass[it] = self.__compute_mass(self.__u[it])
        self.__energy[it] = self.__compute_energy(self.__u[it])

        if not np.isclose(self.__mass[it], self.__mass[0]):
            print(f"\nERROR IN ITERATION: {it:4d}")
            print("MASS IS NOT BEING CONSERVED!!!!!!!!!!!!!!!!!!")
            print(self.__mass[0], self.__mass[it])
            self.__t = self.__t[:it+1]
            self.__mass = self.__mass[:it+1]
            self.__energy = self.__energy[:it+1]
            self.__u = self.__u[:it+1,:]
            return 1
        elif (self.__energy[it]-self.__energy[it-1])/self.__energy[it-1] > 0.01:
            print(f"\nERROR IN ITERATION: {it:4d}")
            print("J INCREASING!!!!!!!!!!!!!!!!!!")
            print(self.__energy[it-1], self.__energy[it], )
            self.__t = self.__t[:it+1]
            self.__mass = self.__mass[:it+1]
            self.__energy = self.__energy[:it+1]
            self.__u = self.__u[:it+1,:]
            return 2
        else:
            return 0
    
    
    
    ######################################################
    # PROPERTIES

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
    