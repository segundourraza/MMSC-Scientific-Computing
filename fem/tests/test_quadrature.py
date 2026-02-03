import unittest
import numpy as np

from fem._quadrature import quadrature

class TestGaussLegendreQuadrature(unittest.TestCase):

    def _poly(self, k):
        return lambda x: x**k

    def _exact_poly_integral(self, k):
        if k % 2 == 1:
            return 0.0
        return 2.0 / (k + 1)

    def test_polynomial_exactness(self):
        """
        n-point Gauss–Legendre integrates polynomials
        of degree ≤ 2n−1 exactly on [-1, 1]
        """
        for n in range(1, 6):
            max_degree = 2 * n - 1
            for k in range(max_degree + 1):
                with self.subTest(n=n, degree=k):
                    numerical = quadrature(self._poly(k), n)
                    exact = self._exact_poly_integral(k)
                    self.assertAlmostEqual(
                        numerical, exact, places=14,
                        msg=f"Exactness failed for n={n}, degree={k}"
                    )

    def test_symmetry_odd_functions(self):
        """Odd functions must integrate to zero on [-1, 1]"""
        odd_functions = [
            lambda x: x,
            lambda x: x**3,
            lambda x: np.sin(x),
        ]

        for n in range(1, 6):
            for f in odd_functions:
                with self.subTest(n=n, f=f):
                    val = quadrature(f, n)
                    self.assertAlmostEqual(val, 0.0, places=14)

    def test_convergence_smooth_function(self):
        """Error should decrease with increasing n"""
        f = lambda x: np.exp(x)
        exact = np.e - 1 / np.e

        errors = []
        for n in range(1, 6):
            numerical = quadrature(f, n)
            errors.append(abs(numerical - exact))

        for i in range(1, len(errors)):
            self.assertLess(
                errors[i], errors[i - 1],
                msg="Convergence with n violated"
            )

if __name__ == '__main__':
    unittest.main()