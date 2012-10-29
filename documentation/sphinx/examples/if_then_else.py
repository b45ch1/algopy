"""
Test piecewise functions.
"""

from numpy.testing import *
from numpy.testing.decorators import *
import numpy

from algopy import *
from algopy.linalg import *
from algopy.special import *

# All of the imports are copied from test_examples.py
# and they put algopy functions like sum and exp in the top level namespace.


def heaviside(x):
    return 0.5*sign(x) + 0.5


def nice_fn_a(x):
    """
    This sine function is OK to evaluate at positive or negative x.
    """
    return sin(x)

def nice_fn_b(x):
    """
    This is a low order Taylor approximation of sin(x) at x=0.
    It is just a polynomial and is well behaved everywhere.
    """
    return x - (x**3) / 6. + (x**5) / 120.

def weird_fn_a(x):
    """
    If you evaluate this function at large positive x you might have trouble.
    """
    return exp(exp(exp(x)))

def weird_fn_b(x):
    """
    If you evaluate this function at negative x you might have a bad time.
    """
    return sqrt(x)


def expit_blend(x, f, g, k=60):
    """
    This function is a smooth blend between functions f and g.
    Well it is not so smooth when k is huge; at huge k it approximates
    the Heaviside step function.
    @param x: independent variable
    @param f: dominant function when x < 0
    @param g: dominant function when 0 < x
    @param k: slope of expit approximation of step function at x=0
    """
    return f(x)*expit(-4*k*x) + g(x)*expit(4*k*x)

def soft_piecewise(x, f, g):
    """
    This is soft in the sense that nans or infs may leak through.
    It is softer than hard_piecewise but not as soft as expit_blend.
    This should be the limit of expit_blend as k approaches infinity.
    @param x: independent variable
    @param f: dominant function when x < 0
    @param g: dominant function when 0 < x
    """
    return f(x)*heaviside(-x) + g(x)*heaviside(x)

def hard_piecewise(x, f, g):
    """
    This function harshly glues together f and g at the origin.
    @param x: independent variable
    @param f: dominant function when x < 0
    @param g: dominant function when 0 < x
    """
    #FIXME: the comparison does not currently vectorize,
    # not that I really expected it to, or necessarily hope that it will.
    # But maybe some kind of replacement can be invented?
    if x <= 0:
        return f(x)
    else:
        return g(x)


class Test_Piecewise(TestCase):

    def test_single_negative_value_nice_fn(self):
        D,P = 4,1
        X = UTPM(numpy.zeros((D,P)))
        X.data[0,0] = -1.23
        X.data[1,0] = 1
        f, g = nice_fn_a, nice_fn_b
        Y = f(X)
        Z = expit_blend(X, f, g)
        W = hard_piecewise(X, f, g)
        V = soft_piecewise(X, f, g)
        assert_allclose(Y.data, Z.data)
        assert_allclose(Y.data, W.data)
        assert_allclose(Y.data, V.data)

    def test_single_positive_value_nice_fn(self):
        D,P = 4,1
        X = UTPM(numpy.zeros((D,P)))
        X.data[0,0] = 2.34
        X.data[1,0] = 1
        f, g = nice_fn_a, nice_fn_b
        Y = g(X)
        Z = expit_blend(X, f, g)
        W = hard_piecewise(X, f, g)
        V = soft_piecewise(X, f, g)
        assert_allclose(Y.data, Z.data)
        assert_allclose(Y.data, W.data)
        assert_allclose(Y.data, V.data)

    def test_single_negative_value_weird_fn(self):
        """
        Note that the expit blend will not work for this example.
        """
        D,P = 4,1
        X = UTPM(numpy.zeros((D,P)))
        X.data[0,0] = -1.23
        X.data[1,0] = 1
        f, g = weird_fn_a, weird_fn_b
        Y = f(X)
        #Z = expit_blend(X, f, g)
        W = hard_piecewise(X, f, g)
        #V = soft_piecewise(X, f, g)
        #assert_allclose(Y.data, Z.data)
        assert_allclose(Y.data, W.data)
        #assert_allclose(Y.data, V.data)

    def test_single_positive_value_weird_fn(self):
        """
        Note that the expit blend will not work for this example.
        """
        D,P = 4,1
        X = UTPM(numpy.zeros((D,P)))
        X.data[0,0] = 2.34
        X.data[1,0] = 1
        f, g = weird_fn_a, weird_fn_b
        Y = g(X)
        #Z = expit_blend(X, f, g)
        W = hard_piecewise(X, f, g)
        #V = soft_piecewise(X, f, g)
        #assert_allclose(Y.data, Z.data)
        assert_allclose(Y.data, W.data)
        #assert_allclose(Y.data, V.data)

    def test_piecewise_positive(self):
        """
        This test does not care whether the hard piecewise vectorizes.
        """
        D,P,N = 4,2,2
        X = UTPM(numpy.random.rand(D,P,N,N))
        X.data[1] = 1.
        f, g = nice_fn_a, nice_fn_b
        Y = expit_blend(X, f, g)
        Z = hard_piecewise(X, f, g)
        V = soft_piecewise(X, f, g)
        assert_allclose(Y.data, Z.data)
        assert_allclose(Y.data, V.data)

    def test_piecewise_negative(self):
        """
        This test does not care whether the hard piecewise vectorizes.
        """
        D,P,N = 4,2,2
        X = UTPM(-numpy.random.rand(D,P,N,N))
        X.data[1] = 1.
        f, g = nice_fn_a, nice_fn_b
        Y = expit_blend(X, f, g)
        Z = hard_piecewise(X, f, g)
        V = soft_piecewise(X, f, g)
        assert_allclose(Y.data, Z.data)
        assert_allclose(Y.data, V.data)

    def test_piecewise_mixed_sign(self):
        """
        This test usually fails, because the hard piecewise does not vectorize.
        """
        D,P,N = 4,2,2
        X = UTPM(2*numpy.random.rand(D,P,N,N)-1)
        X.data[1] = 1.
        f, g = nice_fn_a, nice_fn_b
        Y = expit_blend(X, f, g)
        Z = hard_piecewise(X, f, g)
        V = soft_piecewise(X, f, g)
        assert_allclose(Y.data, Z.data)
        assert_allclose(Y.data, V.data)

    def test_positive_and_negative_weird_functions(self):
        """
        I would like to see a piecewise function that will pass this test.
        The expit blend fails because the nans and infs leak.
        The hard piecewise fails because its decision does not vectorize.
        """
        D = 4
        X = UTPM(numpy.zeros((D,1,2)))
        X.data[0,0,0] = -1.23
        X.data[0,0,1] = 2.34
        X.data[1,0,0] = 1
        X.data[1,0,1] = 1
        f, g = weird_fn_a, weird_fn_b
        Y = UTPM.zeros_like(X)
        Y[0] = f(X[0])
        Y[1] = g(X[1])
        #Z = expit_blend(X, f, g)
        W = hard_piecewise(X, f, g)
        #V = soft_piecewise(X, f, g)
        #assert_allclose(Y.data, Z.data)
        assert_allclose(Y.data, W.data)
        #assert_allclose(Y.data, V.data)


if __name__ == "__main__":
    run_module_suite()

