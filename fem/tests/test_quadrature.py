import unittest, math
import numpy as np

# from .._quadrature import quadrature, triangle_quadrature
from fem._quadrature import quadrature, triangle_quadrature

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



def analytic_monomial_integral_ref_triangle(a, b):
    """
    Analytic integral of x^a y^b over reference triangle (0,0)-(1,0)-(0,1).
    Integral = a! * b! / (a+b+2)!
    """
    return math.factorial(a) * math.factorial(b) / math.factorial(a + b + 2)

class TestTriangleQuadrature(unittest.TestCase):
    def check_rule(self, npts, exact_deg, tol=1e-12):
        pts, w = triangle_quadrature(npts)
        # weights sum to area = 1/2
        wsum = float(np.sum(w))
        self.assertAlmostEqual(wsum, 0.5, places=12,
                               msg=f"weights sum {wsum} != 0.5 for rule {npts}")

        # Check all monomials x^a y^b with total degree <= exact_deg
        for a in range(exact_deg + 1):
            for b in range(exact_deg + 1 - a):
                # evaluate integral by quadrature on reference triangle
                vals = (pts[:, 0] ** a) * (pts[:, 1] ** b)
                q = float(np.dot(w, vals))
                exact = analytic_monomial_integral_ref_triangle(a, b)
                # Use an absolute tolerance scaled slightly for larger magnitude
                # but these integrals are O(1) so a small tol suffices.
                err = abs(q - exact)
                self.assertLessEqual(err, tol,
                                     msg=(f"Rule {npts} failed for monomial x^{a} y^{b}: "
                                          f"quad={q}, exact={exact}, err={err}"))

    def test_1pt_rule(self):
        self.check_rule(npts=1, exact_deg=1, tol=1e-12)

    def test_3pt_rule(self):
        self.check_rule(npts=3, exact_deg=2, tol=1e-12)

    def test_4pt_rule(self):
        self.check_rule(npts=4, exact_deg=3, tol=1e-12)

    def test_6pt_rule(self):
        # degree 4 exactness
        self.check_rule(npts=6, exact_deg=4, tol=1e-12)

    def test_7pt_rule(self):
        # degree 5 exactness
        self.check_rule(npts=7, exact_deg=5, tol=1e-12)

    def test_invalid_npts(self):
        with self.assertRaises(ValueError):
            triangle_quadrature(5)  # unsupported




if __name__ == '__main__':
    unittest.main()