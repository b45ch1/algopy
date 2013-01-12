import warnings

import numpy
import numpy.testing
from numpy.testing import assert_allclose

from algopy import nthderiv

try:
    import sympy
except ImportError as e:
    sympy = None
    raise ImportWarning(e)


class TestLog(numpy.testing.TestCase):

    def test_log_n0(self):
        x = 0.123
        a = nthderiv.log(x, n=0)
        b = numpy.log(x)
        assert_allclose(a, b)

    def test_log_n1(self):
        x = 0.123
        a = nthderiv.log(x, n=1)
        b = 1 / x
        assert_allclose(a, b)

    def test_log_n2(self):
        x = 0.123
        a = nthderiv.log(x, n=2)
        b = -1 / numpy.square(x)
        assert_allclose(a, b)

    def test_log_n3(self):
        x = 0.123
        a = nthderiv.log(x, n=3)
        b = 2 / numpy.power(x, 3)
        assert_allclose(a, b)

    def test_log1p_n0(self):
        x = 0.123
        a = nthderiv.log1p(x, n=0)
        b = numpy.log1p(x)
        assert_allclose(a, b)

    def test_log1p_n1(self):
        x = 0.123
        a = nthderiv.log1p(x, n=1)
        b = 1 / (x + 1)
        assert_allclose(a, b)

    def test_log1p_n2(self):
        x = 0.123
        a = nthderiv.log1p(x, n=2)
        b = -1 / numpy.square(x + 1)
        assert_allclose(a, b)

    def test_log1p_n3(self):
        x = 0.123
        a = nthderiv.log1p(x, n=3)
        b = 2 / numpy.power(x + 1, 3)
        assert_allclose(a, b)



class TestMisc(numpy.testing.TestCase):

    def test_reciprocal(self):
        x = 0.123
        a = nthderiv.reciprocal(x, n=0)
        b = numpy.reciprocal(x)
        assert_allclose(a, b)

    def test_cos(self):
        x = 0.123
        a = nthderiv.cos(x, n=0)
        b = numpy.cos(x)
        assert_allclose(a, b)

    def test_tan(self):
        x = 0.123
        a = nthderiv.tan(x, n=0)
        b = numpy.tan(x)
        assert_allclose(a, b)

    def test_arctan_n0(self):
        x = 0.123
        a = nthderiv.arctan(x, n=0)
        b = numpy.arctan(x)
        assert_allclose(a, b)

    def test_arctan_n1(self):
        x = 0.123
        a = nthderiv.arctan(x, n=1)
        b = 1 / (x*x + 1)
        assert_allclose(a, b)

    def test_cosh(self):
        x = 0.123
        a = nthderiv.cosh(x, n=0)
        b = numpy.cosh(x)
        assert_allclose(a, b)

    def test_sqrt_n0(self):
        x = 0.123
        a = nthderiv.sqrt(x, n=0)
        b = numpy.sqrt(x)
        assert_allclose(a, b)

    def test_sqrt_n1(self):
        x = 0.123
        a = nthderiv.sqrt(x, n=1)
        b = 1 / (2 * numpy.sqrt(x))
        assert_allclose(a, b)

    def test_arccosh_n0(self):
        x = 2.345
        a = nthderiv.arccosh(x, n=0)
        b = numpy.arccosh(x)
        assert_allclose(a, b)

    def test_arccosh_n1(self):
        x = 2.345
        a = nthderiv.arccosh(x, n=1)
        b = 1 / numpy.sqrt(x*x - 1)
        assert_allclose(a, b)

    """
    def test_arccosh_decoration(self):
        print nthderiv.arccosh
        print nthderiv.arccosh.__name__
        print nthderiv.arccosh.__doc__
        print nthderiv.arccosh.domain
        raise Exception
    """


if __name__ == '__main__':
    numpy.testing.run_module_suite()
