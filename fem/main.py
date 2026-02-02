import numpy as np


############################################################################
# BASIS FUNCTIONS
linear_basis_functions = [lambda xi: 0.5*(1-xi),
                          lambda xi: 0.5*(1+xi)]

grad_linear_basis_functions = [lambda xi: -0.5,
                               lambda xi: 0.5]


class LinearElement1D:

    degree: int = 1
    n: int = 2

    @staticmethod
    def basis_functions(xi):
        return (_(xi) for _ in linear_basis_functions)
    
    @staticmethod
    def Me(he):
        return he/6*np.array([[2,1],[1,2]], dtype=float)
    
    @staticmethod
    def Ke(he):
        return 2/he * np.array([[1, -1], [-1, 1]], dtype=float)
    
    @staticmethod
    def InvMe(he):
        return 2/he*np.array([[2, -1],[-1,2]], dtype=float)
    

ELEMENT_MAP = {1: LinearElement1D}


#######################################################################
# TIME INETGRATION
def _explicit_time_integrator(Me, Ke, Ne):
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
    
    
    def l2g_map(self,e,n):
        return self.element.degree*e + n
        
    def solve(self, u0, T:float, dt:float, time_integrator:str|int = 'explicit', non_linear_solver:str|int = 'picard'):
        
        self.__dt = dt
        self.__t = np.arange(dt, T, dt)
        self.__nt = len(self.__t)

        if isinstance(time_integrator, str):
            time_integrator = TIME_INTEGRATOR_STRING2INT_MAP[time_integrator]
        self._step = TIME_INTEGRATOR_MAP[time_integrator]

        if isinstance(non_linear_solver, str):
            non_linear_solver = NL_SOLVER_STRING2INT_MAP[non_linear_solver]
        self._nl_solver = NL_SOLVER_MAP[non_linear_solver]


        self.__solution = np.empty((self.__nt+1, self.__N), dtype=float)
        if callable(u0):
            self.__solution[0,:] = u0(self.x)
        elif len(u0) == self.__N:
            self.__solution[0,:] = u0(self.x)


        ################################################
        # EXPLICIT EULER

        M = self.__assemble_M()
        K = self.__assemble_K()

        for it in range(self.__nt):

           # Compute 
            pass
    
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
    
    def __assemble_N(self, evaluation_c):
        


    def __compute_lhs(self):
        return self.element.Me(he)

        

    @staticmethod
    def _step(self, ):
        """Step solution in time"""
        pass

    @staticmethod
    def _nl_solver(self,):
        """Non linear solver"""
        pass


if __name__ == '__main__':

    # Physics
    dt = 0.01
    tEnd = dt
    epsilon = 0.01

    # Discretization
    L = 3
    N = 3
    poly_degree = 1
    


    # Initial conditions
    def c0(x): return np.cos(np.pi*x)

    # Nonlinear solver


    sol = CahnHilliardSolver(epsilon, 
                             number_of_elements=N, L = L, 
                             polynomial_order=1)
    sol.solve(c0, tEnd, dt, )