import matplotlib.pyplot as plt
import numpy as np
import warnings
import h5py, json
from datetime import datetime, timezone
from pathlib import Path

from scipy.sparse import csc_matrix, bmat
import scipy.sparse.linalg as linalg

from ._config import _progress_range, STABILIZATION_CONSTANT, tqdm
from ._elements import _LegendreElement, LinearLegendreElement, QuadraticLegendreElement





ELEMENT_MAP = {1: LinearLegendreElement(),
               2: QuadraticLegendreElement()}


TIME_INTEGRATOR_STR2INT_MAP = {'explicit': 0, 
                                  'implicit': 1,
                                  'a': 2,
                                  'b': 3, 
                                  '1si': 4,
                                  '1ssi' : 5,
                                  '2ssi' : 6,
                                  }

TIME_INTEGRATOR_INT2STR_MAP = {0: 'explicit', 
                               1: 'implicit',
                               2: 'a',
                               3: 'b',
                               4: '1si',
                               5: '1ssi',
                               6: '2ssi',
                               }

        
class CahnHilliardSolver1D():

    def __init__(self, epsilon:float, number_of_elements: int, L:float, polynomial_order:int = 1):
        
        self.epsilon:float = epsilon
        self.element:_LegendreElement = ELEMENT_MAP[polynomial_order]
        self.__Ne: int = number_of_elements
        self.__N:int = number_of_elements*self.element.degree+1


        self.x = np.linspace(0, L, self.__N)

        # Evaluate 'Mass' and 'Stiffness' matrix. These DO NOT change with time or value of C
        self.M = self.__assemble_M()
        self.K = self.__assemble_K()
        
        
    def solve(self, u0, T:float, dt:float, time_integrator:str|int = 'explicit', nonlinear_solver_options:dict = {},
              terminate_run = True):
        
        self.__T = T
        self.__dt = dt
        self.__t = np.arange(0, T+dt, dt)
        self.__nt = len(self.__t)

        if isinstance(time_integrator, str):
            time_integrator = TIME_INTEGRATOR_STR2INT_MAP[time_integrator.lower()]
        self.__solver_name = TIME_INTEGRATOR_INT2STR_MAP[time_integrator]
        match time_integrator:
            case 0 :
                _time_stepper = self._explicit
            case 1:
                _time_stepper = self._implicit
            case 2:
                _time_stepper = self._semi_implicit_A
            case 3:
                _time_stepper = self._semi_implicit_B
            case 4:
                _time_stepper = self._1SI_scheme
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
            self.__u[0][:self.__N] = u0(self.x)
        elif len(u0) == self.__N:
            self.__u[0][:self.__N] = u0

        # Computation of W[0] via the variational problem
        N0 = self.__assemble_N(self.__u[0,:self.__N])
        b = self.epsilon*(self.K@self.__u[0,:self.__N]) + 1/self.epsilon * N0
        self.__u[0,self.N:] = linalg.spsolve(self.M, b)
        
        # CONSERVED QUANTITIES
        self.__mass = np.zeros((self.__nt,), dtype=float)
        self.__J = np.zeros((self.__nt,), dtype=float)
        self.__mass[0] = self.__compute_mass(self.__u[0])
        self.__J[0] = self.__compute_J(self.__u[0])

        
        ################################################
        # TIME STEPPING

        print()
        _time_stepper(terminate_run)
        tqdm.write("Simulation ended.")



    def __checks(self, it, terminate_run):
        # Compute Conserved quantities
        self.__mass[it] = self.__compute_mass(self.__u[it])
        self.__J[it] = self.__compute_J(self.__u[it])

        if terminate_run:
            if not np.isclose(self.__mass[it], self.__mass[0]):
                print(f"\nERROR IN ITERATION: {it:4d}")
                print("MASS IS NOT BEING CONSERVED!!!!!!!!!!!!!!!!!!")
                print(self.__mass[0], self.__mass[it])
                self.__t = self.__t[:it+1]
                self.__mass = self.__mass[:it+1]
                self.__J = self.__J[:it+1]
                self.__u = self.__u[:it+1,:]
                return 1
            elif (self.__J[it]-self.__J[it-1])/self.__J[it-1] > 0.01:
                print(f"\nERROR IN ITERATION: {it:4d}")
                print("J INCREASING!!!!!!!!!!!!!!!!!!")
                print(self.__J[it-1], self.__J[it], )
                self.__t = self.__t[:it+1]
                self.__mass = self.__mass[:it+1]
                self.__J = self.__J[:it+1]
                self.__u = self.__u[:it+1,:]
                return 2
        return 0

    ####################################################################
    # ASSEMBLE GLOBAL LINEAR SYSTEMS
    def __assemble_M(self)->csc_matrix:
        """Assemble Mass Matrix"""
        M = np.zeros((self.__N, self.__N), dtype= float)
        # Loop over elements
        for e in range(self.__Ne):
            # Classic overlapping block assembly
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            self.element.Me(M[i:i+self.element.n, i:i+self.element.n], he)
        return csc_matrix(M)
    
    def __assemble_K(self)->csc_matrix:
        """Assemble Stiffness Matrix"""
        K = np.zeros((self.__N, self.__N), dtype= float)
        for e in range(self.__Ne):
            # Classic overlapping block assembly
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            self.element.Ke(K[i:i+self.element.n, i:i+self.element.n], he)
        return csc_matrix(K)
    
    def __assemble_N(self, evaluation_C):
        """Assemble non-linear mass matrix"""
        N = np.zeros((self.__N,), dtype=float)
        for e in range(self.__Ne):
            # Classic overlapping block assembly
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            self.element.Ne(N[i:i+self.element.n], he, evaluation_C[i:i+self.element.n])
        return N
    
    def __assemble_H(self, evaluation_C)->csc_matrix:
        """Assemble Nonlinear Jacobian Block"""
        H = np.zeros((self.__N, self.__N), dtype= float)
        for e in range(self.__Ne):
            # Classic overlapping block assembly
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            self.element.He(H[i:i+self.element.n, i:i+self.element.n], he, evaluation_C[i:i+self.element.n])
        return csc_matrix(H)
    
    ########################################################################
    # EXPLICIT TIME STEPPING
    def _explicit(self, terminate_run):
        """Explicit time stepping"""
        # Precompute LU factorisation of Mass matrix
        lu = linalg.splu(self.M)
        for it in _progress_range(range(1,self.__nt), desc = "Simulation running"):
            self.__u[it] = self._step_explicit(self.__u[it-1], lu)

            flag = self.__checks(it, terminate_run=terminate_run)

            if flag != 0:
                break
        
    def _step_explicit(self, u, lu):
        """Explicit step"""
        u_new = np.copy(u)
        Wi = u[self.__N:]
        
        # First Propagate C
        fc = self.M@u[:self.__N] - self.__dt*(self.K@Wi)
        u_new[:self.N] = lu.solve(fc)
        
        # Secondly propagate W with new values of C
        N = self.__assemble_N(u_new[:self.__N])
        fw = self.epsilon*(self.K@u_new[:self.__N]) + 1/self.epsilon*N

        u_new[self.N:] = lu.solve(fw)

        return u_new
    

    ########################################################################
    # IMPLICIT TIME STEPPING
    
    def _implicit(self, terminate_run):
        """Implicit time stepping"""
        def Residual(u_prev, u):
            res = np.zeros((2*self.__N))
            res[:self.__N] = 1/self.__dt * (self.M@(u[:self.__N] - u_prev[:self.__N])) + self.K@u[self.__N:]
            res[self.__N:] = self.M@u[self.__N:] - self.epsilon*(self.K@u[:self.__N]) - (1.0/self.epsilon)*self.__assemble_N(u[:self.__N])
            return res
        
        def Jac(u):
            ck = -self.epsilon*self.K - (1/self.epsilon)*self.__assemble_H(u[:self.__N])
            ck += (1.0/self.epsilon)*self.M
            return bmat([[self.M/self.__dt, self.K],
                        [ck, self.M]], format = 'csc')

        for it in _progress_range(range(1,self.__nt), desc = "Simulation running"):
            self.__u[it] = self._NewtonRaphson(self.__u[it-1], Jac, Residual, **self.__nonlinear_solver_parameters)

            flag = self.__checks(it, terminate_run=terminate_run)
            
            if flag != 0:
                return self.__u[:it-1,:]
    
    
    ########################################################################
    # SEMI-IMPLICIT A TIME STEPPING
    def _semi_implicit_A(self, terminate_run):
        """EYRRE solver, case A"""
        
        def Residual(u_prev, u):
            res = np.zeros((self.__N*2,))
            res[:self.__N] = 1/self.__dt * (self.M@(u[:self.__N] - u_prev[:self.__N])) + self.K@u[self.__N:]
            res[self.__N:] = self.M@u[self.__N:] - self.epsilon*(self.K@u[:self.__N]) \
                - (1.0/self.epsilon)*self._compute_phi1_A(u[:self.__N]) \
                    + (1.0/self.epsilon)*self._compute_phi2_A(u[:self.__N])
            # res[self.__N:] = self.epsilon*(self.K@u[:self.__N]) + (1.0/self.epsilon)*self._compute_phi1_A(u[:self.__N]) - (1.0/self.epsilon)*self._compute_phi2_A(u_prev[:self.__N]) - (self.M@u[self.__N:])
            # res[self.__N:] = self.epsilon*(self.K@u[:self.__N]) + (1.0/self.epsilon)*self._compute_phi1_A(u[:self.__N]) - (1.0/self.epsilon)*self._compute_phi2_A(u_prev[:self.__N]) - (self.M@u[self.__N:])
            return res

        def Jac(u):
            ck = -(self.epsilon)*self.K - (1.0/self.epsilon)*self.__assemble_H(u[:self.__N])
            return bmat([[self.M/self.__dt, self.K],
                        [ck,                self.M]], format = 'csc')

        for it in _progress_range(range(1,self.__nt), desc = "Simulation running"):
            self.__u[it] = self._NewtonRaphson(self.__u[it-1], Jac, Residual, **self.__nonlinear_solver_parameters)
            
            flag = self.__checks(it, terminate_run=terminate_run)


            if flag != 0:
                return self.__u[:it-1,:]


    def _compute_phi1_A(self, evaluation_C):
        b = np.zeros((self.__N,))
        for e in range(self.__Ne):
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            self.element._c3(b[i:i+self.element.n], he, evaluation_C[i:i+self.element.n])    
        return b

    def _compute_phi2_A(self, evaluation_C):
        b = np.zeros((self.__N,))
        for e in range(self.__Ne):
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            self.element._c1(b[i:i+self.element.n], he, evaluation_C[i:i+self.element.n])    
        return b
    
    
    
    ########################################################################
    # SEMI-IMPLICIT B TIME STEPPING
    def _semi_implicit_B(self, terminate_run):
        """Semi-Implicit methods Case B"""

        # Used for Semi-Implicit methods B
        ck = -self.epsilon*self.K - 2/self.epsilon*self.M
        # Compute LHS: This is done once for stencil B
        A = bmat([[self.M/self.__dt, self.K],
                  [ck, self.M]], format='csc')
        b = np.zeros((self.__N*2,))

        lu = linalg.splu(A)
        for it in _progress_range(range(1,self.__nt), desc = "Simulation running"):
            # Update RHS
            self.__update_rhs_B(b, self.__u[it-1,:self.__N])
            
            # SOLVE
            self.__u[it] = lu.solve(b)

            flag = self.__checks(it, terminate_run=terminate_run)
            
            if flag != 0:
                return self.__u[:it-1,:]
        
    def __update_rhs_B(self, b, evaluation_C):
        """Assemble non-linear vector for semi implicit B"""
        b[:self.__N] = 1/self.__dt*(self.M@evaluation_C)
        b[self.__N:] = 0
        for e in range(self.__Ne):
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            self.element.b2_b(b[self.__N+i:self.__N+i+self.element.n], he, evaluation_C[i:i+self.element.n])    
        b[self.__N:] *= 1/self.epsilon

    ########################################################################
    # 1st-order Semi-Implicit Scheme (1SI)

    def _1SI_scheme(self, terminate_run):
        """
        1st-order Semi-Implicit Scheme (1SI)
        
        Ref: NUMERICAL APPROXIMATIONS OF ALLEN-CAHN AND CAHN-HILLIARD EQUATIONS - Jie Shen
        """

        # Compute LHS: This is done once for stencil B
        A = bmat([[-self.epsilon*self.K, self.M],
                  [self.M,               self.__dt*self.K]], format='csc')
        b = np.zeros((self.__N*2,))
        
        # Pre-compute LU factorisation
        lu = linalg.splu(A)
        for it in _progress_range(range(1,self.__nt), desc = "Simulation running"):
            # Update RHS
            self.__update_rhs_1SI(b, self.__u[it-1,:self.__N])
            
            # SOLVE
            self.__u[it] = lu.solve(b)

            flag = self.__checks(it, terminate_run=terminate_run)

            if flag != 0:
                return self.__u[:it-1,:]
        
    def __update_rhs_1SI(self, b, evaluation_C):
        """Assemble non-linear vector for semi implicit B"""
        temp = (self.M@evaluation_C)
        b[self.__N:] = temp
        b[:self.__N] = -temp
        for e in range(self.__Ne):
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            self.element._c3(b[i:i+self.element.n], he, evaluation_C[i:i+self.element.n])
        b[:self.__N] *= 1/self.epsilon

    
    
    ########################################################################
    # 1st-order Stabilized Semi-Implicit Scheme (1SSI)

    def _1SSI_scheme(self, terminate_run):
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
        for it in _progress_range(range(1,self.__nt), desc = "Simulation running"):
            # Update RHS
            self.__update_rhs_1SSI(b, self.__u[it-1,:self.__N], STABILIZATION_CONSTANT)
            
            # SOLVE
            self.__u[it] = lu.solve(b)
            # self.__u[it] = linalg.cg(A, b)

            flag = self.__checks(it, terminate_run=terminate_run)

            if flag != 0:
                return self.__u[:it-1,:]
        
    def __update_rhs_1SSI(self, b, evaluation_C, S):
        """Assemble non-linear vector for 1st order stabilized semi-implicit scheme"""
        temp = (self.M@evaluation_C)
        b[self.__N:] = temp
        b[:self.__N] = -temp - S*temp
        for e in range(self.__Ne):
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            self.element._c3(b[i:i+self.element.n], he, evaluation_C[i:i+self.element.n])
        b[:self.__N] *= 1/self.epsilon



    
    ########################################################################
    # 2nd order Stabilized Semi-Implicit Scheme (2SSI)

    def _2SSI_scheme(self, terminate_run):
        """
        2nd-order Stabilized Semi-Implicit Scheme (2SSI)
        
        Ref: NUMERICAL APPROXIMATIONS OF ALLEN-CAHN AND CAHN-HILLIARD EQUATIONS - Jie Shen
        """
    
        # USE A TIME STEP OF 1SSI
        # Since the scheme is second order, we require two previous computations to propagate solution
        # Compute LHS: This is done once for stencil B
        A11 = -self.epsilon*self.K - STABILIZATION_CONSTANT/self.epsilon*self.M
        A = bmat([[A11 ,    self.M],
                  [self.M,  self.__dt*self.K]], format='csc')
        b = np.zeros((self.__N*2,))
        
        # Pre-compute LU factorisation
        lu = linalg.splu(A)
        # Update RHS
        self.__update_rhs_1SSI(b, self.__u[0,:self.__N], STABILIZATION_CONSTANT)
        # Solve
        self.__u[1] = lu.solve(b)
        # Run checks
        flag = self.__checks(1, terminate_run=terminate_run)


        # START USING 2SSI  
        A11 = -self.epsilon*self.K - STABILIZATION_CONSTANT/self.epsilon*self.M
        A = bmat([[A11 ,    self.M],
                  [self.M,  2/3*self.__dt*self.K]], format='csc')
        lu = linalg.splu(A)
        
        for it in _progress_range(range(2,self.__nt), desc = "Simulation running"):
            # Update RHS
            self.__update_rhs_2SSI(b, self.__u[it-2,:self.__N], self.__u[it-1,:self.__N], STABILIZATION_CONSTANT)
            
            # SOLVE
            self.__u[it] = lu.solve(b)

            flag = self.__checks(it, terminate_run=terminate_run)
            
            if flag != 0:
                return self.__u[:it-1,:]
        
    def __update_rhs_2SSI(self, b, evaluation_C1, evaluation_C2, S):
        """Assemble non-linear vector for 1st order stabilized semi-implicit scheme"""
        # Mc1 = (self.M@evaluation_C1)
        # Mc2 = (self.M@evaluation_C2)
        
        # phi1 = np.zeros((self.__N,))
        # phi2 = np.zeros((self.__N,))
        # phi1[:] = -Mc1
        # phi2[:] = -Mc2
        # for e in range(self.__Ne):
        #     i = e*self.element.degree
        #     he = self.x[i+self.element.n-1] - self.x[i]    
        #     self.element._c3(phi1[i:i+self.element.n], he, evaluation_C1[i:i+self.element.n])
        #     self.element._c3(phi2[i:i+self.element.n], he, evaluation_C2[i:i+self.element.n])
        # b[:self.__N] = -2*S*Mc2 + S*Mc1 + 2*(phi2) - phi1
        # b[:self.__N] *= 1/self.epsilon
        # b[self.__N:] = 4/3*Mc2 - 1/3*Mc1

        phi1 = np.zeros((self.__N,))
        phi2 = np.zeros((self.__N,))
        for e in range(self.__Ne):
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]    
            self.element._c3(phi1[i:i+self.element.n], he, evaluation_C1[i:i+self.element.n])
            self.element._c3(phi2[i:i+self.element.n], he, evaluation_C2[i:i+self.element.n])
        b[:self.__N] = -2*(STABILIZATION_CONSTANT + 1)*self.M@evaluation_C2 + 2*phi2 +\
                          (STABILIZATION_CONSTANT + 1)*self.M@evaluation_C1 - phi1
        b[:self.__N] *= 1/self.epsilon
        b[self.__N:] = self.M@(1/3*(4*evaluation_C2 - evaluation_C1))




    ########################################################################
    # NONLINEAR SOLVER
    def fd_jacobian_check(self, Residual, Jac, u_prev, u, eps=1e-6):
        Fu = Residual(u_prev, u)
        J = Jac(u).tocsc()
        v = np.random.randn(u.size); v /= np.linalg.norm(v)
        Jv = J.dot(v)
        FD = (Residual(u_prev, u + eps*v) - Fu) / eps
        res = np.linalg.norm(Jv - FD) / (np.linalg.norm(Jv) + 1e-16)
        if res < eps or np.isclose(res, eps):
            return res
        else:
            raise RuntimeError(f"Finite Difference Jacobian Check Failed. eps ({eps}) != residual ({res})")

    def _NewtonRaphson(self, u_prev, Jac, Residual,
                       tol = 1e-8, max_iter = 100,
                       line_search = None, relaxation_parameter = 0,
                       verbose = False, run_checks = False):
        """
        Newton solver for one implicit time-step.
        u_prev : vector at previous time step (size 2*N)
        Returns u_new (size 2*N)
        """
        # start from previous solution as initial guess
        u = np.copy(u_prev)
        error = []
        
        if relaxation_parameter < 0 or relaxation_parameter >=1.0:
            raise ValueError(f"'relaxation_parameter' must be on range (0, 1), currently equal to {relaxation_parameter}.")
        
        if isinstance(line_search, str):
            match line_search.lower():
                case 'armijo':
                    update_rule = lambda u_prev, u, du, res_norm: self.apply_backtracking(u_prev, u, du, res_norm,Residual, relaxation_parameter=relaxation_parameter)[0]
                case _: raise ValueError("'line_search' must be on eof {'armijo'}, currently {}".format(line_search))
        elif line_search is None:
            update_rule = lambda u_prev, u, du, res_norm: u + (1-relaxation_parameter)*du 
        else:
            raise ValueError("Unrecognized 'line_search' algorithim")
        
        
        for it in range(max_iter):
            # build residual at current iterate
            res = Residual(u_prev, u)
            res_norm = np.linalg.norm(res)
            
            if verbose:
                error.append(res_norm)
                print(f"  Newton iteration: {it}, ||R(u)|| = {res_norm:.3e}")
                        
            # convergence check
            if res_norm < tol:
                if verbose:
                    print()
                # if run_checks:
                #     fig, ax = plt.subplots()
                #     ax.semilogy(error)
                #     plt.show()
                return u

            # assemble Jacobian for current u
            J = Jac(u[:self.__N])

            if run_checks:
                j_res= self.fd_jacobian_check(Residual, Jac, u_prev, u)
                print(f"\t Jacobian relative error = {j_res:.4e}")

            # Solve linear system J * du = -res
            # factorize for speed/stability
            lu = linalg.splu(J)
            du = lu.solve(-res)

            # update
            u = update_rule(u_prev, u, du, res_norm)
            
        # if we exit loop not converged:
        warnings.warn("Newton-Raphson failed to converge after max_iter")
        return u

    def apply_backtracking(self, u_prev, u, du, res_norm, Residual, relaxation_parameter = 0.0, max_iters=10, c=1e-4, rho=0.5):
        """Backtracking Armijo line-search. Returns new u, alpha used."""
        alpha = 1.0 - relaxation_parameter
        Fu = lambda v: np.linalg.norm(Residual(u_prev, v))  # adjust if your Residual needs different args
        f0 = res_norm
        for k in range(max_iters):
            u_trial = u + alpha * du
            f_trial = Fu(u_trial)
            if f_trial <= f0 + c * alpha * (-np.dot(Residual(u_prev, u), du)):  # Armijo condition
                return u_trial, alpha
            alpha *= rho
        # if line search fails, return the damped update
        return u + alpha * du, alpha


    ####################################################################
    # AUXILIARY FUNCTIONS
    def __compute_mass(self, u):
        M = 0
        for e in range(self.__Ne):
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            M += self.element.compute_mass_e(he, u[i:i+self.element.n])
        return M
    
    def __compute_J(self, u):
        J = 0    
        for e in range(self.__Ne):
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            J += self.element.compute_J_e(self.epsilon, he, u[i:i+self.element.n])
        return J
    
    def plot_contour(self, levels = 100, cmap = 'jet'):
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(self.x, self.t)
        C = ax.contourf(X, Y, self.sol_c, levels = levels, cmap = cmap)
        fig.colorbar(C)
        ax.set_xlabel('x')
        ax.set_ylabel('t', rotation = 0, labelpad= 10)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    def interpolate_elements(self,u, nodes_per_elements:int):

        x_interp = np.ones((self.__Ne*(nodes_per_elements-1)+1,))*99
        c_interp = np.zeros((self.__Ne*(nodes_per_elements-1)+1,))
        w_interp = np.zeros((self.__Ne*(nodes_per_elements-1)+1,))


        for e in range(self.Ne):
            i = e*self.element.degree
            xiq = np.linspace(-1, 1, num=nodes_per_elements)
            phi = self.element.basis_functions(xiq)
            j = e*(nodes_per_elements-1) 
            
            x_interp[j:j+nodes_per_elements] = np.dot(self.x[i:i+self.element.n], phi)
            c_interp[j:j+nodes_per_elements] = np.dot(u[i:i+self.element.n], phi)
            w_interp[j:j+nodes_per_elements] = np.dot(u[self.__N+i:self.__N+i+self.element.n], phi)
        return x_interp, c_interp, w_interp

    def save(self, filename = None, directory = None, append_time = False):
        if filename is None:
            filename = f"Cahn_Hilliard_solution_{self.__solver_name}_CG{self.element.degree}_Ne{self.Ne}_T{self.__T:.1e}_dt{self.__dt:.1e}"

        if append_time:
            filename += "_" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")

        # Ensure extension
        if not filename.endswith(".h5"):
            filename += ".h5"
        
        # Determine directory
        if directory is None:
            directory = Path.cwd() / "solution"
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
            for name, array in zip(['sol_c', 'sol_w', 't', 'x', 'mass', 'J'], [self.sol_c, self.sol_w, self.t, self.x, self.__mass, self.__J]):
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
        
        print("\nSimulation successfully saved as : {}".format(filepath))
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
    def J(self):
        """J integrals"""
        return self.__J
    