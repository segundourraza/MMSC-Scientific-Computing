import numpy as np
np.set_printoptions(linewidth = 240)
from _quadrature import quadrature, GAUSS_QUADRATURE_POINTS, GAUSS_QUADRATURE_WEIGHTS

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


    @staticmethod
    def basis_functions(xi):
        return [_(xi) for _ in linear_basis_functions]
    
    @staticmethod
    def Me(he):
        return he/6*np.array([[2,1],[1,2]], dtype=float)
    
    @staticmethod
    def Ke(he):
        return 2/he * np.array([[1, -1], [-1, 1]], dtype=float)
    
    @staticmethod
    def InvMe(he):
        return 2/he*np.array([[2, -1],[-1,2]], dtype=float)
    
    def Ne(self, he, Ce):
        Ne = np.empty((self.n, self.n))
        for xi,wi in zip(GAUSS_QUADRATURE_POINTS[self.r_Ne], GAUSS_QUADRATURE_WEIGHTS[self.r_Ne]):
            phi0, phi1 = self.basis_functions(xi)
            ch2 = (Ce[0]*phi0 + Ce[1]*phi1)**2
            for i in range(self.n):
                for j in range(self.n):
                    Ne[i,j] += ch2*phi0*phi1*he/2*wi
        return Ne

    def Fnle(self, he, C0):
        Fnl = np.empty((self.n,))
        for xi,wi in zip(GAUSS_QUADRATURE_POINTS[self.r_Ne], GAUSS_QUADRATURE_WEIGHTS[self.r_Ne]):
            phi = self.basis_functions(xi)
            ch = (C0[0]*phi[0] + C0[1]*phi[1])
            fprime = ch**3 - ch
            for i in range(self.n):
                Fnl[i] += wi*fprime*phi[i]*(he/2)
        return Fnl


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
    
        
    def solve_transient(self, u0, T:float, dt:float, time_integrator:str|int = 'explicit', non_linear_solver:str|int = 'picard'):
        
        self.__dt = dt
        self.__t = np.arange(0, T+dt, dt)
        self.__nt = len(self.__t)

        if isinstance(time_integrator, str):
            time_integrator = TIME_INTEGRATOR_STRING2INT_MAP[time_integrator]
        self._step = TIME_INTEGRATOR_MAP[time_integrator]

        if isinstance(non_linear_solver, str):
            non_linear_solver = NL_SOLVER_STRING2INT_MAP[non_linear_solver]
        self._nl_solver = NL_SOLVER_MAP[non_linear_solver]


        
        # Evaluate 'Mass' and 'Stiffness' matrix. These DO NOT change with time or value of C
        M = self.__assemble_M()
        invM = self.__assemble_InvM()
        K = self.__assemble_K()


        # Solution vector u = [C , W]
        self.__u = np.empty((self.__nt, 2*self.__N), dtype=float)
        if callable(u0):
            self.__u[0][:self.__N] = u0(self.x)
        elif len(u0) == self.__N:
            self.__u[0][:self.__N] = u0
        
        


        ################################################
        # EXPLICIT EULER
        a = 1/self.epsilon
        invMK = invM @ K


        # Explicit Euler Requires the computation of W[0] via the variational problem
        Fnl = np.zeros((self.__N,))
        for e in range(self.__ne):
            i = e*self.element.degree
            he = self.x[self.l2g_map(e, self.element.degree)] - self.x[self.l2g_map(e, 0)]
            Fnl[i:i+self.element.n] += self.element.Fnle(he, self.__u[0][i:i+self.element.n])
        self.__u[0][self.N:] = self.epsilon*invMK@self.__u[0][:self.N] + 1/self.epsilon*Fnl

        

        for it in range(1,self.__nt):
            Ci = self.__u[it-1][:self.__N]
            Wi = self.__u[it-1][self.__N:]
            
            N = self.__assemble_N(Ci)
            # Compute RHS vector
            f = -a*Ci  + self.epsilon*(invMK@Ci) + a*(invM@(N@Ci))

            # Compute new C^{i+1}
            self.__u[it][:self.N] = Ci - dt*invMK@Wi
            self.__u[it][self.N:] = f

            print(self.__u[it])
            print('\n')
        
        return self.__u



    ####################################################################
    # AUXILIARY FUNCTIONS

    def __assemble_M(self):
        """Assemble Mass Matrix"""
        M = np.zeros((self.__N, self.__N), dtype= float)
        # Loop over elements
        for e in range(self.__ne):
            # Classic overlapping block assembly
            i = e*self.element.degree
            he = self.x[self.l2g_map(e, self.element.degree)] - self.x[self.l2g_map(e, 0)]
            M[i:i+self.element.n, i:i+self.element.n] += self.element.Me(he)
        return M
    
    def __assemble_InvM(self):
        """Assemble the inverse of the Mass Matrix"""
        InvM = np.zeros((self.__N, self.__N), dtype= float)
        # Loop over elements
        for e in range(self.__ne):
            # Classic overlapping block assembly
            i = e*self.element.degree
            he = self.x[self.l2g_map(e, self.element.degree)] - self.x[self.l2g_map(e, 0)]
            InvM[i:i+self.element.n, i:i+self.element.n] += self.element.InvMe(he)
        return InvM

    def __assemble_K(self):
        """Assemble Stiffness Matrix"""
        K = np.zeros((self.__N, self.__N), dtype= float)
        # Loop over elements
        for e in range(self.__ne):
            # Classic overlapping block assembly
            i = e*self.element.degree
            he = self.x[self.l2g_map(e, self.element.degree)] - self.x[self.l2g_map(e, 0)]
            K[i:i+self.element.n, i:i+self.element.n] += self.element.Ke(he)
        return K
    
    def __assemble_N(self, evaluation_C):
        """Assemble non-linear mass matrix"""
        N = np.zeros((self.__N,), dtype=float)
        for e in range(self.__ne):
            i = e*self.element.degree
            he = self.x[self.l2g_map(e, self.element.degree)] - self.x[self.l2g_map(e, 0)]
            N[i:i+self.element.n] += self.element.Ne(he, evaluation_C[i:i+self.element.n])
        return N


    @staticmethod
    def _step(self, ):
        """Step solution in time"""
        pass

    @staticmethod
    def _nl_solver(self,):
        """Non linear solver"""
        pass



    #######################################################
    # HELPER FUNCTIONS
    def l2g_map(self,e,n):
        return self.element.degree*e + n




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


if __name__ == '__main__':

    # Physics
    dt = 1e-4
    tEnd = dt*3
    epsilon = 0.01

    # Discretization
    L = 1
    N = 10
    poly_degree = 1
    


    # Initial conditions
    def c0(x): return np.cos(np.pi*x)
    def c0(x): return np.sin(np.pi/(L)*x)

    # Nonlinear solver


    sol = CahnHilliardSolver(epsilon, 
                             number_of_elements=N, L = L, 
                             polynomial_order=1)
    sol.solve_transient(c0, tEnd, dt)
    

    import matplotlib.pyplot as plt
    
    plt.plot(sol.x, c0(sol.x))
    for i in range(1, sol.nt):
        plt.plot(sol.x, sol.sol_c[i])
    plt.show()