import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from scipy.sparse import csc_matrix, bmat
import scipy.sparse.linalg as linalg


from ._elements1D import _LegendreElement, LinearLegendreElement, QuadraticLegendreElement





ELEMENT_MAP = {1: LinearLegendreElement(),
               2: QuadraticLegendreElement()}


TIME_INTEGRATOR_STRING2INT_MAP = {'explicit': 0, 
                                  'implicit': 1,}



class CahnHilliardSolver():

    def __init__(self, epsilon:float, number_of_elements: int, L:float, polynomial_order:int = 1):
        
        self.epsilon:float = epsilon
        self.element:_LegendreElement = ELEMENT_MAP[polynomial_order]
        self.__ne: int = number_of_elements
        self.__N:int = number_of_elements*self.element.degree+1


        self.x = np.linspace(0, L, self.__N)

        # Evaluate 'Mass' and 'Stiffness' matrix. These DO NOT change with time or value of C
        self.M = self.__assemble_M()
        self.K = self.__assemble_K()
        
        
    def solve(self, u0, T:float, dt:float, time_integrator:str|int = 'explicit', non_linear_solver:str|int = 'picard'):
        
        self.__dt = dt
        self.__t = np.arange(0, T+dt, dt)
        self.__nt = len(self.__t)

        if isinstance(time_integrator, str):
            time_integrator = TIME_INTEGRATOR_STRING2INT_MAP[time_integrator]
        match time_integrator:
            case 0 :
                _time_stepper = self._explicit
            case 1:
                _time_stepper = self._implicit
            case 2:
                _time_stepper = self._semi_implicit_B
            case _:
                raise ValueError

        # CONSTRUCT SOLUTION VECTOR U = [C , W]
        self.__u = np.empty((self.__nt, 2*self.__N), dtype=float)
        if callable(u0):
            self.__u[0][:self.__N] = u0(self.x)
        elif len(u0) == self.__N:
            self.__u[0][:self.__N] = u0

        # Computation of W[0] via the variational problem
        N0 = self.__assemble_N(self.__u[0,:self.__N])
        b = self.epsilon*(self.K@self.__u[0,:self.__N]) + 1/self.epsilon * N0
        self.__u[0,self.N:] = linalg.spsolve(self.M, b)
        
        # CONSERVED QUANTITIES
        self.__mass = np.empty((self.__nt,), dtype=float)
        self.__J = np.empty((self.__nt,), dtype=float)
        self.__mass[0] = self.__compute_mass(self.__u[0])
        self.__J[0] = self.__compute_J(self.__u[0])

        
        ################################################
        # TIME STEPPING
        _time_stepper()



    def __checks(self, it):
        # Compute Conserved quantities
        self.__mass[it] = self.__compute_mass(self.__u[it])
        self.__J[it] = self.__compute_J(self.__u[it])

        if not np.isclose(self.__mass[it], self.__mass[it-1]):
            print("\n\nMASS IS NOT BEING CONSERVED!!!!!!!!!!!!!!!!!!")
            print(self.__mass[it-1], self.__mass[it])
            self.__t = self.__t[:it]
            self.__mass = self.__mass[:it]
            self.__J = self.__J[:it]
            self.__u = self.__u[:it,:]
            return 1
        elif (self.__J[it]-self.__J[it-1])/self.__J[it-1] > 0.05:
            print("\n\nJ INCREASING!!!!!!!!!!!!!!!!!!")
            print(self.__J[it-1], self.__J[it], )
            self.__t = self.__t[:it]
            self.__mass = self.__mass[:it]
            self.__J = self.__J[:it]
            self.__u = self.__u[:it,:]
            return 2
        else:
            return 0

    ####################################################################
    # ASSEMBLE GLOBAL LINEAR SYSTEMS
    def __assemble_M(self)->csc_matrix:
        """Assemble Mass Matrix"""
        M = np.zeros((self.__N, self.__N), dtype= float)
        # Loop over elements
        for e in range(self.__ne):
            # Classic overlapping block assembly
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            M[i:i+self.element.n, i:i+self.element.n] += self.element.Me(he)
        return csc_matrix(M)
    
    def __assemble_K(self)->csc_matrix:
        """Assemble Stiffness Matrix"""
        K = np.zeros((self.__N, self.__N), dtype= float)
        # Loop over elements
        for e in range(self.__ne):
            # Classic overlapping block assembly
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            K[i:i+self.element.n, i:i+self.element.n] += self.element.Ke(he)
        return csc_matrix(K)
    
    def __assemble_N(self, evaluation_C):
        """Assemble non-linear mass matrix"""
        N = np.zeros((self.__N,), dtype=float)
        for e in range(self.__ne):
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            N[i:i+self.element.n] += self.element.Ne(he, evaluation_C[i:i+self.element.n])
        return N
    
    def __assemble_Jacobian(self)->csc_matrix:
        test = self.__assemble_H(self.__u[0,:self.__N])
        return bmat([[self.M/self.__dt, self.K],
                     [test, self.M]], format = 'csc')
    
    def __assemble_H(self, evaluation_C)->csc_matrix:
        """Assemble Nonlinear Jacobian Block"""
        H = np.zeros((self.__N, self.__N), dtype= float)
        # Loop over elements
        for e in range(self.__ne):
            # Classic overlapping block assembly
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            H[i:i+self.element.n, i:i+self.element.n] += self.element.He(he, evaluation_C[i:i+self.element.n])
        return csc_matrix(H)
    
    ########################################################################
    # EXPLICIT TIME STEPPING
    def _explicit(self):
        """Explicit time stepping"""
        # Precompute LU factorisation of Mass matrix
        lu = linalg.splu(self.M)
        for it in tqdm(range(1,self.__nt)):
            self.__u[it] = self._step_explicit(self.__u[it-1], lu)

            flag = self.__checks(it)

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
    
    def _implicit(self):
        """Implicit time stepping"""
        # Compute Constant blocks of Jacobian
        Jac = self.__assemble_Jacobian()
        
        # for it in tqdm(range(1,self.__nt)):
        for it in range(1,self.__nt):
            self.__u[it] = self._NewtonRaphson(self.__u[it-1])

            flag = self.__checks(it)

            if flag != 0:
                return self.__u[:it-1,:]
    
    
    ########################################################################
    # SEMI-IMPLICIT TIME STEPPING
    def _semi_implicit_B(self):
        """Semi-Implicit methods CASE B"""

        # Used for Semi-Implicit methods B
        ck = -self.epsilon*self.K - 2/self.epsilon*self.M
        A = bmat([[self.M/self.__dt, self.K],
                  [ck, self.M]], format='csc')
        b = np.zeros((self.__N*2,))

        lu = linalg.splu(A)
        for it in range(1,self.__nt):
            # Compute RHS
            b[:self.__N] = 1/self.__dt*(self.M@self.__u[it-1,:self.N])
            b[self.__N:] = 1/self.epsilon*self.__assemble_b2(self.__u[it-1,:self.__N])
            
            # Compute LHS: THis can be done once
            
            # SOLVE
            self.__u[it] = lu.solve(b)

            flag = self.__checks(it)

            if it%100 == 0:
                print(f'{it:4}',end = '\r')

            if flag != 0:
                return self.__u[:it-1,:]
        
    def __assemble_b2(self, evaluation_C):
        """Assemble non-linear mass matrix"""
        b2 = np.zeros((self.__N,), dtype=float)
        for e in range(self.__ne):
            i = e*self.element.degree
            he = self.x[i+self.element.n-1] - self.x[i]
            b2[i:i+self.element.n] += self.element.b2(he, evaluation_C[i:i+self.element.n])
        return b2


    ########################################################################
    # NONLINEAR SOLVER
    def fd_jacobian_check(self, u, eps=1e-6):
        Fu = self.Residual(u, self.__u[0])
        J = self.Jac(u).tocsc()
        v = np.random.randn(u.size); v /= np.linalg.norm(v)
        Jv = J.dot(v)
        FD = (self.Residual(u + eps*v, self.__u[0]) - Fu) / eps
        rel_err = np.linalg.norm(Jv - FD) / (np.linalg.norm(Jv) + 1e-16)
        print("rel_err =", rel_err)
    
    def Jac(self, u):
        ck = -self.epsilon*self.K - (1/self.epsilon)*self.__assemble_H(u[:self.__N])
        return bmat([[self.M/self.__dt, self.K],
                     [ck, self.M]], format = 'csc')

    def Residual(self, u, u0):
        Rc = 1/self.__dt * (self.M@(u[:self.__N] - u0[:self.__N])) + self.K@u[self.__N:]
        Rw = self.M@u[self.__N:] - self.epsilon*(self.K@u[:self.__N]) - 1/self.epsilon*self.__assemble_N(u[:self.__N])
        return np.concatenate([Rc, Rw])




    def _NewtonRaphson(self, u0, tol = 1e-8, max_iter = 50, verbose = False):
        """Newton Raphson solver"""
        u1 = np.copy(u0)
        error = []
        for i in range(max_iter):
            # Compute Residuals
            Rc = 1/self.__dt * (self.M@(u1[:self.__N] - u0[:self.__N])) + self.K@u1[self.__N:]
            Rw = self.M@u1[self.__N:] - self.epsilon*(self.K@u1[:self.__N]) - 1/self.epsilon*self.__assemble_N(u1[:self.__N])
            # Rw = self.M@u1[self.__N:] - self.epsilon**2*(self.K@u1[:self.__N]) - self.__assemble_N(u1[:self.__N])
            
            # Compute Jacobian and LHS
            # ck = -self.epsilon*self.K - (1/self.epsilon)*self.__assemble_H(u1[:self.__N])
            # J[self.__N:, :self.__N] = ck
            J = self.Jac(u1)
            b = np.concatenate([Rc, Rw])
            
            # Solve system
            du = linalg.spsolve(J, -b)
            
            # Update u
            u1 += du
            
            test = np.linalg.norm(b)            
            error.append(test)
            if verbose:
                print('\niteration:', i)
                print('\t', u1[:self.__N])
                print('\t', u1[self.__N:])
                print()
                print('\t',Rc)
                print('\t',Rw)
                print()
                print('\t',du[:self.__N])
                print('\t',du[self.__N:])
                print()
                if i == 0:
                    print(f'\ttest: {test:.4e}')
                else:
                    print(f'\ttest: {test:.4e}', f'\tRate of Convergence: {test/old_test**2:.1e}')
            if test < tol:
                # error = np.array(error)
                # plt.plot(range(i+1), np.log(error))
                # idx = np.arange(i-5, i+1, dtype=int)
                # print(np.polyfit(idx, np.log(error[idx]), 1)[0])
                # plt.show()        
                # self.fd_jacobian_check(u1)
                return u1
            old_test = test
            
        else:
            # error = np.array(error)
            # plt.plot(range(i+1), np.log(error))
            # idx = np.arange(i-5, i+1)
            # print(np.polyfit(idx, np.log(error[idx]), 1)[0])
            # plt.show()        
            
            raise RuntimeError("Newton-Rapshon failed to converge")


    def _NewtonRaphson_New(self, u_prev, tol = 1e-8, max_iter = 50, verbose = True):
        """
        Newton solver for one implicit time-step.
        u_prev : vector at previous time step (size 2*N)
        Returns u_new (size 2*N)
        """
        # start from previous solution as initial guess
        u = np.copy(u_prev)

        for it in range(max_iter):
            # build residual at current iterate
            Rc = (1.0/self.__dt) * (self.M @ (u[:self.__N] - u_prev[:self.__N])) + self.K @ u[self.__N:]
            Rw = (self.M @ u[self.__N:]) - self.epsilon*(self.K @ u[:self.__N]) - (1.0/self.epsilon) * self.__assemble_N(u[:self.__N])
            res = np.concatenate([Rc, Rw])
            res_norm = np.linalg.norm(res)
            
            if verbose:
                print(f"  Newton it {it}, ||res|| = {res_norm:.3e}")

            # convergence check
            if res_norm < tol:
                return u

            # assemble Jacobian for current u
            H = self.__assemble_H(u[:self.__N])                           # H depends on u
            ck = - self.epsilon * self.K - (1.0 / self.epsilon) * H

            J = bmat([[self.M / self.__dt,      self.K],
                    [ck,                     self.M]], format='csc')

            # Solve linear system J * du = -res
            # factorize for speed/stability
            lu = linalg.splu(J)
            du = lu.solve(-res)

            # update
            u += du

            # damping / safety: if update is huge, optionally damp (not necessary for small dt)
            if np.linalg.norm(du) > 1e-1 * np.linalg.norm(u):
                u -= 0.5*du   # example damping

        # if we exit loop not converged:
        raise RuntimeError("Newton-Raphson failed to converge after max_iter")


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
    
    def plot_contour(self, levels = 100, cmap = 'hot'):
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(self.x, self.t)
        C = ax.contourf(X, Y, self.sol_c, levels = levels, cmap = cmap)
        fig.colorbar(C)
        ax.set_xlabel('x')
        ax.set_ylabel('t', rotation = 0, labelpad= 10)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    def interpolate_elements(self,u, nodes_per_elements:int):

        x_interp = np.ones((self.__ne*(nodes_per_elements-1)+1,))*99
        c_interp = np.zeros((self.__ne*(nodes_per_elements-1)+1,))
        w_interp = np.zeros((self.__ne*(nodes_per_elements-1)+1,))


        for e in range(self.Ne):
            i = e*self.element.degree
            xiq = np.linspace(-1, 1, num=nodes_per_elements)
            phi = self.element.basis_functions(xiq)
            j = e*(nodes_per_elements-1) 
            
            x_interp[j:j+nodes_per_elements] = np.dot(self.x[i:i+self.element.n], phi)
            c_interp[j:j+nodes_per_elements] = np.dot(u[i:i+self.element.n], phi)
            w_interp[j:j+nodes_per_elements] = np.dot(u[self.__N+i:self.__N+i+self.element.n], phi)
        return x_interp, c_interp, w_interp


    
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
    