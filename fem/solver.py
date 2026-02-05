import numpy as np
import scipy.linalg as linalg
np.set_printoptions(linewidth = 240)

############################################################################
# BASIS FUNCTIONS
linear_basis_functions = [lambda xi: 0.5*(1-xi),
                          lambda xi: 0.5*(1+xi)]

grad_linear_basis_functions = [lambda xi: -0.5,
                               lambda xi: 0.5]


class LinearElement1D:

    degree: int = 1
    n: int = 2
    
    # Quadrature points
    r_Ne:int = 2
    r_J: int = 3


    @staticmethod
    def basis_functions(xi):
        return [_(xi) for _ in linear_basis_functions]
    
    @staticmethod
    def grad_basis_functions(xi):
        return [_(xi) for _ in grad_linear_basis_functions]
    
    @staticmethod
    def Me(he):
        return he/6*np.array([[2,1],[1,2]], dtype=float)
    
    @staticmethod
    def Ke(he):
        return 1/he * np.array([[1, -1], [-1, 1]], dtype=float)

    def Ne(self, he, Ce):
        Ne = np.zeros((self.n))
        for xi, wi in zip(*np.polynomial.legendre.leggauss(self.r_Ne)):
            phi = self.basis_functions(xi)
            ch = Ce[0]*phi[0] + Ce[1]*phi[1]
            ch3 = (ch)**3
            for i in range(self.n):
                Ne[i] += (ch3 - ch)*phi[i]*he/2*wi
        return Ne
    

    def compute_mass_e(self, he, Ce):
        return he/2*(Ce[0] + Ce[1])
    
    def compute_J_e(self, eps, he, Ce):
        J = 0
        for xi, wi in zip(*np.polynomial.legendre.leggauss(self.r_J)):
            phi = self.basis_functions(xi)
            grad_phi = self.grad_basis_functions(xi)
            ch = Ce[0]*phi[0] + Ce[1]*phi[1]
            grad_ch = Ce[0]*grad_phi[0] + Ce[1]*grad_phi[1]
            J += wi*((1 - ch**2)**2/(4*eps)  + eps/2 * (grad_ch)**2)
        return J*he/2
        

ELEMENT_MAP = {1: LinearElement1D()}


#######################################################################
# TIME INETGRATION
def _explicit_time_integrator(dt, u, A, B, f):
    "Solves A x u^{i+1} = B x u^{i} + dt x f"


    pass



TIME_INTEGRATOR_STRING2INT_MAP = {'explicit': 0}
TIME_INTEGRATOR_MAP = {0: _explicit_time_integrator}





#######################################################################
# NONLINEAR SOLVERS

def _picard_iteration():
    """Solve non-linear system of equations via picard iteration"""
    pass

NL_SOLVER_STRING2INT_MAP = {'picard': 0}
NL_SOLVER_MAP = {0: _picard_iteration}








class CahnHilliardSolver():

    def __init__(self, epsilon:float, number_of_elements: int, L:float, polynomial_order:int = 1):
        
        self.epsilon:float = epsilon
        self.element = ELEMENT_MAP[polynomial_order]
        self.__ne: int = number_of_elements
        self.__N:int = number_of_elements+1


        self.x = np.linspace(0, L, self.__N)

        # Evaluate 'Mass' and 'Stiffness' matrix. These DO NOT change with time or value of C
        self.M = self.__assemble_M()
        self.K = self.__assemble_K()
        
        
    def solve_transient(self, u0, T:float, dt:float, time_integrator:str|int = 'explicit', non_linear_solver:str|int = 'picard'):
        
        self.__dt = dt
        self.__t = np.arange(0, T+dt, dt)
        self.__nt = len(self.__t)

        if isinstance(time_integrator, str):
            time_integrator = TIME_INTEGRATOR_STRING2INT_MAP[time_integrator]
        _step = TIME_INTEGRATOR_MAP[time_integrator]

        if isinstance(non_linear_solver, str):
            non_linear_solver = NL_SOLVER_STRING2INT_MAP[non_linear_solver]
        self._nl_solver = NL_SOLVER_MAP[non_linear_solver]

        # CONSTRUCT SOLUTION VECTOR U = [C , W]
        self.__u = np.empty((self.__nt, 2*self.__N), dtype=float)
        if callable(u0):
            self.__u[0][:self.__N] = u0(self.x)
        elif len(u0) == self.__N:
            self.__u[0][:self.__N] = u0
        
        # Computation of W[0] via the variational problem
        N0 = self.__assemble_N(self.__u[0,:self.__N])
        b = self.epsilon*(self.K@self.__u[0,:self.__N]) + 1/self.epsilon * N0
        self.__u[0,self.N:] = linalg.solve(self.M, b)
        
        # CONSERVED QUANTITIES
        self.__mass = np.empty((self.__nt,), dtype=float)
        self.__J = np.empty((self.__nt,), dtype=float)
        self.__mass[0] = self.__compute_mass(self.__u[0])
        self.__J[0] = self.__compute_J(self.__u[0])

        ################################################
        # TIME STEPPING
        
        # Precompute LU factorisation of Mass matrix
        lu, piv = linalg.lu_factor(self.M)
        
        
        for it in range(1,self.__nt):
            
            self.__u[it] = _step(self.__u[it-1], (lu, piv))

            # Compute Conserved quantities
            self.__mass[it] = self.__compute_mass(self.__u[it])
            self.__J[it] = self.__compute_J(self.__u[it])

            if not np.isclose(self.__mass[it], self.__mass[it-1]):
                print("\n\nMASS IS NOT BEING CONSERVED!!!!!!!!!!!!!!!!!!")
                print(self.__mass[it-1], self.__mass[it])
                return self.__u[:it-1,:]
            
            if (self.__J[it]-self.__J[it-1])/self.__J[it-1] > 0.05:
                print("\n\nJ INCREASING!!!!!!!!!!!!!!!!!!")
                print(self.__J[it-1], self.__J[it], )
                return self.__u[:it-1,:]
            
            print(self.__mass[it])
            print(self.__J[it])
            print()        
        return self.__u



    ####################################################################
    # ASSEMBLE GLOBAL LINEAR SYSTEMS
    def __assemble_M(self):
        """Assemble Mass Matrix"""
        M = np.zeros((self.__N, self.__N), dtype= float)
        # Loop over elements
        for e in range(self.__ne):
            # Classic overlapping block assembly
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            M[i:i+self.element.n, i:i+self.element.n] += self.element.Me(he)
        return M
    
    def __assemble_K(self):
        """Assemble Stiffness Matrix"""
        K = np.zeros((self.__N, self.__N), dtype= float)
        # Loop over elements
        for e in range(self.__ne):
            # Classic overlapping block assembly
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            K[i:i+self.element.n, i:i+self.element.n] += self.element.Ke(he)
        return K
    
    def __assemble_N(self, evaluation_C):
        """Assemble non-linear mass matrix"""
        N = np.zeros((self.__N,), dtype=float)
        for e in range(self.__ne):
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            N[i:i+self.element.n] += self.element.Ne(he, evaluation_C[i:i+self.element.n])
        return N
    

    ####################################################################
    # AUXILIARY FUNCTIONS
    def __compute_mass(self, u):
        M = 0
        for e in range(self.__ne):
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            M += self.element.compute_mass_e(he, u[i:i+self.element.n])
        return M
    
    def __compute_J(self, u):
        J = 0    
        for e in range(self.__ne):
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            J += self.element.compute_J_e(self.epsilon, he, u[i:i+self.element.n])
        return J
    

    ########################################################################
    # TIME STEPPING
    def _step_explicit(self, u):
        """Explicit time stepping"""
        u_new = np.empty((2*self.__N,), dtype=float)
        Wi = u[self.__N:]
        
        # First Propagate C
        fc = self.M@u[:self.__N] - self.__dt*(self.K@Wi)
        u_new[:self.N] = linalg.lu_solve((lu, piv), fc)
        
        # Secondly propagate W with new values of C
        N = self.__assemble_N(u_new[:self.__N])
        fw = self.epsilon*(self.K@u_new[:self.__N]) + 1/self.epsilon*N
        u_new[self.N:] = linalg.lu_solve((lu, piv), fw)

        return u_new


    ########################################################################
    # NONLINEAR SOLVER

    @staticmethod
    def _nl_solver(self,):
        """Non linear solver"""
        pass



    
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
        return self.__ne
        
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
    def mass(self):
        """Mass"""
        return self.__mass
    
    
    @property
    def J(self):
        """J integrals"""
        return self.__J