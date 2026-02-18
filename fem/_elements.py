from abc import ABC, abstractmethod
from ._quadrature import triangle_quadrature
import numpy as np

############################################################################
# BASIS FUNCTIONS
linear_basis_functions = [lambda xi: 0.5*(1-xi),
                          lambda xi: 0.5*(1+xi)]

linear_grad_basis_functions = [lambda xi: -0.5,
                               lambda xi: 0.5]

quad_basis_function = [lambda x: -0.5*x*(1-x), 
                       lambda x: 1 - x*x, 
                       lambda x: 0.5*x*(1+x)]

quad_grad_basis_function = [lambda x: -0.5*(1-2*x), 
                            lambda x: -2*x, 
                            lambda x: 0.5*(1+2*x)]



class _LegendreElement(ABC):

    @property
    @abstractmethod
    def degree(self)->int:
        """Degree of polynomial"""
        pass

    @property
    def n(self)->int:
        """Number of node in element"""
        return self.degree + 1
    
    @property
    @abstractmethod
    def r_Ne(self)->int:
        """Quadrature points used for Ne"""
        pass

    @property
    @abstractmethod
    def r_J(self)->int:
        """Quadrature points used for Jacobian Nonlinear block"""
        pass

    
    @staticmethod
    @abstractmethod
    def basis_functions(xi):
        """Basis functions"""
        pass
    
    @staticmethod
    @abstractmethod
    def grad_basis_functions(xi):
        """Basis functions"""
        pass
    
    @staticmethod
    @abstractmethod
    def Me(he):
        """Mass Matrix"""
        pass
    
    @staticmethod
    @abstractmethod
    def Ke(he):
        """Stiffness Matrix"""
        pass
    

    def Ne(self, N, he, Ce):
        for xi, wi in zip(*np.polynomial.legendre.leggauss(self.r_Ne)):
            phi = self.basis_functions(xi)
            ch = np.dot(Ce, phi)
            ch3 = (ch)**3
            for i in range(self.n):
                N[i] += (ch3 - ch)*phi[i]*(he/2)*wi
    
    def He(self, H, he, Ce):
        for xi, wi in zip(*np.polynomial.legendre.leggauss(self.r_Ne)):
            phi = self.basis_functions(xi)
            ch2 = np.dot(Ce, phi)**2
            H[:] += np.outer(phi, phi)*(3*ch2)*(he/2)*wi

    #################################
    # TIME STEPPING
    def b2_b(self, b2, he, Ce):
        for xi, wi in zip(*np.polynomial.legendre.leggauss(self.r_Ne)):
            phi = self.basis_functions(xi)
            ch = np.dot(Ce, phi)
            ch3 = (ch)**3
            for i in range(self.n):
                b2[i] += (ch3 - 3*ch)*phi[i]*(he/2)*wi


    def _c1(self, b, he, Ce):
        for xi, wi in zip(*np.polynomial.legendre.leggauss(self.r_Ne)):
            phi = self.basis_functions(xi)
            ch = np.dot(Ce, phi)
            for i in range(self.n):
                b[i] += ch*phi[i]*(he/2)*wi

    def _c3(self, b, he, Ce):
        for xi, wi in zip(*np.polynomial.legendre.leggauss(self.r_Ne)):
            phi = self.basis_functions(xi)
            ch3 = np.dot(Ce, phi)**3
            for i in range(self.n):
                b[i] += ch3*phi[i]*(he/2)*wi
    
    
    def Awc_c(self, H, he, Ce):
        for xi, wi in zip(*np.polynomial.legendre.leggauss(self.r_Ne)):
            phi = self.basis_functions(xi)
            ch2 = np.dot(Ce, phi)**2
            H[:] += 3*ch2*np.outer(phi, phi)*(he/2)*wi

    
    #################################
    # AUXILIARY 
    
    def compute_J_e(self, eps, he, Ce):
        J = 0
        for xi, wi in zip(*np.polynomial.legendre.leggauss(self.r_J)):
            phi = self.basis_functions(xi)
            grad_phi = self.grad_basis_functions(xi)
            ch = np.dot(Ce,phi)
            grad_ch = np.dot(Ce, grad_phi)
            J += wi*((1 - ch**2)**2/(4*eps)  + eps/2 * (grad_ch)**2)
        return J*he/2

    def compute_mass_e(self, he, Ce):
        M = 0
        for xi, wi in zip(*np.polynomial.legendre.leggauss(self.degree)):
            phi = self.basis_functions(xi)
            ch = np.dot(Ce, phi)
            M += wi*(he/2)*ch
        return M
    
class LinearLegendreElement(_LegendreElement):

    degree: int = 1
    
    # Quadrature points
    r_Ne:int = 3
    r_J: int = 3


    @staticmethod
    def basis_functions(xi):
        return [_(xi) for _ in linear_basis_functions]
    
    @staticmethod
    def grad_basis_functions(xi):
        return [_(xi) for _ in linear_grad_basis_functions]
    

    ##################################################
    # FINITE ELEMENT DISCRETIZATION
    @staticmethod
    def Me(M, he):
        M[:] += he/6*np.array([[2,1],[1,2]], dtype=float)
    
    @staticmethod
    def Ke(K, he):
        K[:] += 1/he * np.array([[1, -1], [-1, 1]], dtype=float)

    ############################################
    # AUXILIARY
    def compute_mass_e(self, he, Ce):
        return he/2*(Ce[0] + Ce[1])

class QuadraticLegendreElement(_LegendreElement):

    degree: int = 2
    
    # Quadrature points
    r_Ne:int = 5
    r_J: int = 5


    @staticmethod
    def basis_functions(xi):
        return [_(xi) for _ in quad_basis_function]
    
    @staticmethod
    def grad_basis_functions(xi):
        return [_(xi) for _ in quad_grad_basis_function]
    

    ##################################################
    # FINITE ELEMENT DISCRETIZATION
    @staticmethod
    def Me(M, he):
        M[:] += he/30*np.array([[ 4, 2, -1],
                                [ 2, 16, 2],
                                [-1, 2,  4]], dtype=float)
    
    @staticmethod
    def Ke(K, he):
        K[:] += 1/(3*he)*np.array([[7,  -8,  1],
                                   [-8, 16, -8],
                                   [1,  -8,  7]], dtype = float)


class LinearTriangularElement():


    n = 3 # Number of nodes in element

    # Quadrature points
    r_Ne:int = 6

    

    #############################################
    # BASIC ELEMENT MATRICES
    __A = 0.5*np.array([[1, -1, 0],
                        [-1, 1, 0],
                        [0, 0, 0]], dtype=float)
    
    __BpC = 0.5*np.array([[2, -1, -1],
                          [-1, 0, 1],
                          [-1, 1, 0]], dtype=float)
    
    __D = 0.5*np.array([[1, 0, -1],
                        [0, 0, 0],
                        [-1, 0, 1]], dtype=float)
    
    __M = (1/24)*np.array([[2, 1, 1],
                           [1, 2, 1],
                           [1, 1, 2]], dtype=float)
    
    def Me(self, M_global, con, detJ):
        M_global[np.ix_(con,con)] += detJ*self.__M
    
    def Ke(self, K_global, con,detJ, invJ):
        Se = (invJ[0,0]**2)*self.__A +(invJ[0,0]*invJ[0,1])*self.__BpC + (invJ[0,1]**2)*self.__D
        Sn = (invJ[1,0]**2)*self.__A +(invJ[1,0]*invJ[1,1])*self.__BpC + (invJ[1,1]**2)*self.__D
        K_global[np.ix_(con,con)] += detJ*(Se + Sn)

    def Ne(self, N, con, detJ, Ce):
        for (xi,eta), wi in zip(*triangle_quadrature(self.r_Ne)):
            phi = self.basis_functions(xi,eta)
            ch = np.dot(Ce, phi)
            ch3 = (ch)**3
            N[con] += (ch3 - ch)*phi*detJ*wi
    
    
    ################################################################
    # TIME STEPPING SPECIFIC MATRICES

        
    def b2_b(self, b_global, con, detJ, Ce):
        for (xi, eta), wi in zip(*triangle_quadrature(self.r_Ne)):
            phi = self.basis_functions(xi, eta)
            ch = np.dot(Ce, phi)
            ch3 = (ch)**3
            b_global[con] += (ch3 - 3*ch)*phi*(detJ)*wi

    def _c3(self, b_global, con, detJ, Ce):
        for (xi, eta), wi in zip(*triangle_quadrature(self.r_Ne)):
            phi = self.basis_functions(xi, eta)
            ch3 = np.dot(Ce, phi)**3
            b_global[con] += ch3*phi*(detJ)*wi


    def compute_energy(self, detJ, invJ, Ce, eps):
        E = 0
        for (xi, eta), wi in zip(*triangle_quadrature(6)):
            phi = self.basis_functions(xi, eta)
            ch2 = np.dot(Ce, phi)**2

            grad_phi = self.grad_basis_function(xi, eta)
            
            dcdx = np.dot(Ce, [[invJ[0,0]*grad_phi[0][0] + invJ[0,1]*grad_phi[0][1]],
                               [invJ[0,0]*grad_phi[1][0] + invJ[0,1]*grad_phi[1][1]],
                               [invJ[0,0]*grad_phi[2][0] + invJ[0,1]*grad_phi[2][1]]])**2
            dcdy = np.dot(Ce, [[invJ[1,0]*grad_phi[0][0] + invJ[1,1]*grad_phi[0][1]],
                               [invJ[1,0]*grad_phi[1][0] + invJ[1,1]*grad_phi[1][1]],
                               [invJ[1,0]*grad_phi[2][0] + invJ[1,1]*grad_phi[2][1]]])**2

            grad_c_norm2 = dcdx + dcdy
            E += (1/(4*eps)*(1- ch2**2)**2 + eps/2*grad_c_norm2)*(detJ)*wi
        return E




    @staticmethod
    def basis_functions(xi, eta):
        return np.array([1 - xi - eta, xi, eta], dtype =float)
    
    @staticmethod
    def grad_basis_function(xi, eta):
        return np.array([[-1, -1],
                         [1, 0],
                         [0, 1]], dtype=float)
    
    @staticmethod
    def compute_mass_e(area, Ce):
        return area*np.sum(Ce)/3
    


