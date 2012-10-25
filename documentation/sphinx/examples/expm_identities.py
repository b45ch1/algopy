"""
Test higher order derivatives of expm using some identities.

These tests only deal with special cases of expm.
They require the matrix to be symmetric.
They only test higher order derivatives with respect to a single
scaling parameter, as opposed to parameters that affect the matrix
in more complicated ways.
"""

from numpy.testing import *
from numpy.testing.decorators import *
import numpy

from algopy import *
from algopy.linalg import *

# All of the imports are copied from test_examples.py
# and they put algopy functions like sum and exp in the top level namespace.


def expm_tS_v1(t, S):
    """
    Compute expm(t*S) using AlgoPy default expm.
    t: truncated univariate Taylor polynomial
    S: symmetric numpy matrix
    """

    # Compute the expm using the default Taylor-aware implementation.
    # As of the time of writing this comment, this default implementation
    # uses a fixed-order Pade approximation.
    return expm(t * S)


def expm_tS_v2(t, S):
    """
    Compute expm(t*S) using AlgoPy eigendecomposition and an identity.
    t: truncated univariate Taylor polynomial
    S: symmetric numpy matrix
    """

    # Compute the eigendecomposition using a Taylor-aware eigh.
    L, Q = eigh(t * S)

    return dot(Q * exp(L), Q.T)


def expm_tS_v3(t, S):
    """
    Compute expm(t*S) using LAPACK eigendecomposition and an identity.
    t: truncated univariate Taylor polynomial
    S: symmetric numpy matrix
    """

    # Compute the eigendecomposition using a Taylor-naive eigh.
    L, Q = numpy.linalg.eigh(S)

    return dot(Q * exp(t * L), Q.T)


def create_random_symmetric_matrix():
    S_asym = numpy.random.rand(4, 4)
    return S_asym + S_asym.T


class Test_ExpmScaledSymmetric(TestCase):

    def test_d0(self):
        S = create_random_symmetric_matrix()
        t0 = 0.123
        t = t0
        raw_v1 = expm_tS_v1(t, S)
        raw_v2 = expm_tS_v2(t, S)
        raw_v3 = expm_tS_v3(t, S)
        assert_allclose(raw_v1, raw_v2)
        assert_allclose(raw_v1, raw_v3)

    def test_d1(self):
        S = create_random_symmetric_matrix()
        t0 = 0.123
        t_grad = UTPM.init_jacobian(t0)
        raw_v1 = expm_tS_v1(t_grad, S)
        raw_v2 = expm_tS_v2(t_grad, S)
        raw_v3 = expm_tS_v3(t_grad, S)
        assert_allclose(raw_v1.data, raw_v2.data)
        assert_allclose(raw_v1.data, raw_v3.data)
        grad_v1 = UTPM.extract_jacobian(sum(raw_v1))
        grad_v2 = UTPM.extract_jacobian(sum(raw_v2))
        grad_v3 = UTPM.extract_jacobian(sum(raw_v3))
        assert_allclose(grad_v1, grad_v2)
        assert_allclose(grad_v1, grad_v3)

    def test_d2(self):
        S = create_random_symmetric_matrix()
        t0 = 0.123
        t_hess = UTPM.init_hessian(t0)
        raw_v1 = expm_tS_v1(t_hess, S)
        raw_v2 = expm_tS_v2(t_hess, S)
        raw_v3 = expm_tS_v3(t_hess, S)
        assert_allclose(raw_v1.data, raw_v2.data)
        assert_allclose(raw_v1.data, raw_v3.data)
        hess_v1 = UTPM.extract_hessian(1, sum(raw_v1))
        hess_v2 = UTPM.extract_hessian(1, sum(raw_v2))
        hess_v3 = UTPM.extract_hessian(1, sum(raw_v3))
        assert_allclose(hess_v1, hess_v2)
        assert_allclose(hess_v1, hess_v3)


if __name__ == "__main__":
    run_module_suite()

