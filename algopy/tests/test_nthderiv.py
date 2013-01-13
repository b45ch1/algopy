"""
Test the nthderiv module.

Note that some of the numdifftools documentation is out of date.
To figure out the right way to use numdifftools I had to directly inspect
http://code.google.com/p/numdifftools/source/browse/trunk/numdifftools/core.py
Also note that comparisons that use relative tolerances,
such as are used by assert_allclose by default,
do not work well when you are comparing to zero.
"""

import functools
import warnings

import numpy as np
import numpy.testing
from numpy.testing import assert_allclose, assert_equal

from algopy import nthderiv

try:
    import sympy
except ImportError as e:
    warnings.warn('some tests require the sympy package')
    sympy = None

try:
    import numdifftools
except ImportError as e:
    warnings.warn('some tests require the numdifftools package')
    numdifftools = None

# an example list of simple x values to be used for testing
g_simple_xs = [
        0.135,
        -0.567,
        1.1234,
        -1.23,
        ]

# an example list of more complicated x values to be used for testing
g_complicated_xs = g_simple_xs + [
        np.array([[0.123, 0.2], [0.93, 0.44]]),
        ]

def assert_allclose_or_small(a, b, rtol=1e-7, zerotol=1e-7):
    if np.amax(np.abs(a)) > zerotol or np.amax(np.abs(b)) > zerotol:
        numpy.testing.assert_allclose(a, b, rtol=rtol)

def gen_named_functions():
    for name, f in nthderiv.__dict__.items():
        domain = getattr(f, 'domain', None)
        extras = getattr(f, 'extras', None)
        if domain is not None and extras is not None:
            yield name, f


class TestAuto(numpy.testing.TestCase):

    def test_syntax(self):
        for name, f in gen_named_functions():
            print
            print name
            # some of the x values are outside of the domain of some functions
            valid_xs = [x for x in g_complicated_xs if np.all(f.domain(x))]
            for x in valid_xs:
                print 'x:', x
                # some of the functions need extra parameters
                args = [1] * f.extras + [x]
                for n in range(4):
                    print 'n:', n
                    ya = f(*args, n=n)
                    print ya
                    assert_equal(np.shape(x), np.shape(ya))
                    yb = np.empty_like(x)
                    f(*args, out=yb, n=n)
                    assert_equal(ya, yb)

    @numpy.testing.decorators.skipif(numdifftools is None)
    def test_numdifftools(self):
        imprecise_functions = [
                nthderiv.hyp2f0,
                nthderiv.hyp3f0,
                ]
        for name, f in gen_named_functions():
            if f in imprecise_functions:
                continue
            print
            print name
            # some of the x values are outside of the domain of some functions
            valid_xs = [x for x in g_simple_xs if f.domain(x)]
            for x in valid_xs:
                print 'x:', x
                # some of the functions need extra parameters
                extra_args = [1] * f.extras
                args = extra_args + [x]
                for n in range(1, 5):
                    print 'n:', n
                    ya = f(*args, n=n)
                    print 'ya:', ya
                    f_part = functools.partial(f, *extra_args)
                    yb = numdifftools.Derivative(f_part, n=n)(x)
                    print 'yb:', yb
                    # we are only detecting gross errors
                    assert_allclose_or_small(ya, yb, rtol=1e-2, zerotol=1e-2)


class TestLog(numpy.testing.TestCase):

    def test_log_n0(self):
        x = 0.123
        a = nthderiv.log(x, n=0)
        b = np.log(x)
        assert_allclose(a, b)

    def test_log_n1(self):
        x = 0.123
        a = nthderiv.log(x, n=1)
        b = 1 / x
        assert_allclose(a, b)

    def test_log_n2(self):
        x = 0.123
        a = nthderiv.log(x, n=2)
        b = -1 / np.square(x)
        assert_allclose(a, b)

    def test_log_n3(self):
        x = 0.123
        a = nthderiv.log(x, n=3)
        b = 2 / np.power(x, 3)
        assert_allclose(a, b)

    def test_log1p_n0(self):
        x = 0.123
        a = nthderiv.log1p(x, n=0)
        b = np.log1p(x)
        assert_allclose(a, b)

    def test_log1p_n1(self):
        x = 0.123
        a = nthderiv.log1p(x, n=1)
        b = 1 / (x + 1)
        assert_allclose(a, b)

    def test_log1p_n2(self):
        x = 0.123
        a = nthderiv.log1p(x, n=2)
        b = -1 / np.square(x + 1)
        assert_allclose(a, b)

    def test_log1p_n3(self):
        x = 0.123
        a = nthderiv.log1p(x, n=3)
        b = 2 / np.power(x + 1, 3)
        assert_allclose(a, b)



class TestMisc(numpy.testing.TestCase):

    def test_reciprocal(self):
        x = 0.123
        a = nthderiv.reciprocal(x, n=0)
        b = np.reciprocal(x)
        assert_allclose(a, b)

    def test_cos(self):
        x = 0.123
        a = nthderiv.cos(x, n=0)
        b = np.cos(x)
        assert_allclose(a, b)

    def test_tan(self):
        x = 0.123
        a = nthderiv.tan(x, n=0)
        b = np.tan(x)
        assert_allclose(a, b)

    def test_arctan_n0(self):
        x = 0.123
        a = nthderiv.arctan(x, n=0)
        b = np.arctan(x)
        assert_allclose(a, b)

    def test_arctan_n1(self):
        x = 0.123
        a = nthderiv.arctan(x, n=1)
        b = 1 / (x*x + 1)
        assert_allclose(a, b)

    def test_cosh(self):
        x = 0.123
        a = nthderiv.cosh(x, n=0)
        b = np.cosh(x)
        assert_allclose(a, b)

    def test_sqrt_n0(self):
        x = 0.123
        a = nthderiv.sqrt(x, n=0)
        b = np.sqrt(x)
        assert_allclose(a, b)

    def test_sqrt_n1(self):
        x = 0.123
        a = nthderiv.sqrt(x, n=1)
        b = 1 / (2 * np.sqrt(x))
        assert_allclose(a, b)

    def test_arccosh_n0(self):
        x = 2.345
        a = nthderiv.arccosh(x, n=0)
        b = np.arccosh(x)
        assert_allclose(a, b)

    def test_arccosh_n1(self):
        x = 2.345
        a = nthderiv.arccosh(x, n=1)
        b = 1 / np.sqrt(x*x - 1)
        assert_allclose(a, b)


if __name__ == '__main__':
    numpy.testing.run_module_suite()
