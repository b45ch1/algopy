"""
Univariate nth derivatives of several numpy and scipy functions.

These functions are intended for two purposes.
They can be used for testing,
and they can also be used as components of
unsophisticated implementations of more complicated functions.
The functions in this module do not support complex numbers
as input or output types, although some of the functions use them internally.
Sometimes scipy is used for sqrt because it can deal with negative numbers.
"""


import math

import numpy as np
import numpy.testing
from numpy.testing import assert_allclose

try:
    import scipy
    import scipy.special
except ImportError:
    pass

__all__ = [
        'rint', 'fix', 'floor', 'ceil', 'trunc',
        'sign', 'absolute',
        'erf', 'erfi', 'gammaln', 'psi', 'polygamma',
        'exp', 'exp2', 'expm1',
        'log', 'log2', 'log10', 'log1p',
        'sqrt', 'square', 'negative', 'reciprocal',
        'sin', 'cos', 'tan',
        'arcsin', 'arccos', 'arctan',
        'sinh', 'cosh', 'tanh',
        'arcsinh', 'arccosh', 'arctanh',
        ]




##############################################################################
# These constants and functions are not defined in numpy or in scipy.

np_sqrt_pi = np.sqrt(np.pi)

def np_sech(x, out=None):
    return np.reciprocal(np.cosh(x), out)

def np_coth(x, out=None):
    return np.reciprocal(np.tanh(x), out)

def np_csch(x, out=None):
    return np.reciprocal(np.sinh(x), out)

def np_real(z, out=None):
    """
    Most numpy functions support an 'out' argument but np.real does not.
    """
    if out is None:
        return np.real(z)
    else:
        out[...] = np.real(z)
        return out

def np_filled_like(x, fill_value, out=None):
    if out is None:
        out = np.copy(x)
    out.fill(fill_value)
    return out

def np_recip_sqrt(x, out=None):
    return np.reciprocal(scipy.sqrt(x), out)


##############################################################################
# Functions that are not so differentiable.

def rint(n, x, out=None):
    if n == 0:
        return np.rint(x, out)
    else:
        return np_filled_like(x, 0, out)

def fix(n, x, out=None):
    if n == 0:
        return np.fix(x, out)
    else:
        return np_filled_like(x, 0, out)

def floor(n, x, out=None):
    if n == 0:
        return np.floor(x, out)
    else:
        return np_filled_like(x, 0, out)

def ceil(n, x, out=None):
    if n == 0:
        return np.ceil(x, out)
    else:
        return np_filled_like(x, 0, out)

def trunc(n, x, out=None):
    if n == 0:
        return np.rint(x, out)
    else:
        return np_filled_like(x, 0, out)

def sign(n, x, out=None):
    if n == 0:
        return np.sign(x, out)
    else:
        return np_filled_like(x, 0, out)

def absolute(n, x, out=None):
    if n == 0:
        return np.absolute(x, out)
    elif n == 1:
        return np.sign(x, out)
    else:
        return np_filled_like(x, 0, out)


##############################################################################
# Miscellaneous special functions.

def erf(n, x, out=None):
    if n == 0:
        return scipy.special.erf(x, out)
    else:
        a = pow(2, -n) * math.factorial(n - 1) * np.reciprocal(np_sqrt_pi)
        b = np.exp(-np.square(x))
        c = np.zeros_like(x)
        for k in range(1, n+1):
            s_a = pow(-1, k-1) * pow(2, 2*k) * pow(x, 2*k - n - 1)
            s_b = math.factorial(2*k - n - 1) * math.factorial(n - k)
            c += s_a * s_b
        return np.multiply(a*b, c, out)

def erfi(n, x, out=None):
    if n == 0:
        return scipy.special.erfi(x, out)
    else:
        a = pow(2, -n) * math.factorial(n - 1) * np.reciprocal(np_sqrt_pi)
        b = np.exp(np.square(x))
        c = np.zeros_like(x)
        for k in range(1, n+1):
            s_a = pow(2, 2*k) * pow(x, 2*k - n - 1)
            s_b = math.factorial(2*k - n - 1) * math.factorial(n - k)
            c += s_a * s_b
        return np.multiply(a*b, c, out)

def gammaln(n, x, out=None):
    if n == 0:
        return scipy.special.gammaln(x, out)
    else:
        return scipy.special.polygamma(n-1, x, out)

def psi(n, x, out=None):
    return scipy.special.polygamma(n, x, out)

def polygamma(m, n, x, out=None):
    return scipy.special.polygamma(m+n, x, out)


##############################################################################
# Exponential, log, power, and polynomial functions.

def exp(n, x, out=None):
    return np.exp(x, out)

def exp2(n, x, out=None):
    out = np.exp2(x, out)
    out *= pow(np.log(2), n)
    return out

def expm1(n, x, out=None):
    if n == 0:
        return np.expm1(x, out)
    else:
        return np.exp(x, out)

def log(n, x, out=None):
    if n == 0:
        return np.log(x, out)
    else:
        out = np.power(x, -n, out)
        out *= pow(-1, n-1) * math.factorial(n-1)
        return out

def log2(n, x, out=None):
    out = log(n, x, out)
    out /= np.log(2)
    return out

def log10(n, x, out=None):
    out = log(n, x, out)
    out /= np.log(10)
    return out

def log1p(n, x, out=None):
    if n == 0:
        return np.log1p(x, out)
    else:
        out = np.power(1 + x, -n, out)
        out *= pow(-1, n-1) * math.factorial(n-1)
        return out

def sqrt(n, x, out=None):
    out = np.power(x, 0.5 - n, out)
    out *= scipy.special.poch(1.5 - n, n)
    return out

def square(n, x, out=None):
    if n == 0:
        return np.square(x, out)
    elif n == 1:
        return np.multiply(x, 2, out)
    elif n == 2:
        return np_filled_like(x, 2, out)
    else:
        return np_filled_like(x, 0, out)

def negative(n, x, out=None):
    if n == 0:
        return np.negative(x, out)
    elif n == 1:
        return np_filled_like(x, -1, out)
    else:
        return np_filled_like(x, 0, out)

def reciprocal(n, x, out=None):
    out = np.power(x, -(n+1), out)
    out *= math.factorial(n) * pow(-1, n)
    return out


##############################################################################
# Trigonometric functions and their functional inverses.

def sin(n, x, out=None):
    return np.sin(0.5 * n * np.pi + x, out)

def cos(n, x, out=None):
    return np.cos(0.5 * n * np.pi + x, out)

def tan(n, x, out=None):
    if out is None:
        out = np.empty_like(x)
    out.fill(0)
    if n == 0:
        out += np.tan(x)
    if n == 1:
        out += np.square(np.sec(x))
    for k in range(n):
        for j in range(k):
            c = pow(np.cos(x), -2*k - 2)
            s = np.sin(0.5 * n * np.pi + 2*(k - j)*x)
            p = pow(-1, k) * pow(k-j, n-1) * pow(2, n - 2*k)
            b = scipy.special.binom(n-1, k) * scipy.special.binom(2*k, j)
            out += n * (c * s * p * b) / (k + 1)
    return out

def arcsin(n, x, out=None):
    if n == 0:
        return np.arcsin(x, out)
    else:
        x1 = np_recip_sqrt(1 - np.square(x))
        a = 1j * pow(-1j, n) * math.factorial(n-1) * pow(x1, n)
        b = scipy.special.eval_legendre(n-1, 1j * x * x1)
        return np_real(a*b, out)

def arccos(n, x, out=None):
    if n == 0:
        return np.arccos(x, out)
    else:
        return np.negative(arcsin(n, x), out)

def arctan(n, x, out=None):
    if n == 0:
        return np.arctan(x, out)
    else:
        a = 0.5j * pow(-1, n) * math.factorial(n - 1)
        b = pow(x - 1j, -n) - pow(x + 1j, -n)
        return np_real(a*b, out)


##############################################################################
# Hyperbolic trigonometric functions and their functional inverses.

def sinh(n, x, out=None):
    return np_real(pow(-1j, n) * np.sinh(0.5j * np.pi * n + x), out)

def cosh(n, x, out=None):
    return np_real(pow(-1j, n) * np.cosh(0.5j * np.pi * n + x), out)

def tanh(n, x, out=None):
    y = np.zeros_like(x, dtype=complex)
    if n == 0:
        y += np.tanh(x)
    if n == 1:
        y += np.square(np_sech(x))
    for k in range(n):
        for j in range(k):
            c = pow(np.cosh(x), -2*k - 2)
            s = np.sinh(0.5j * n * np.pi + 2*(k - j)*x)
            p = pow(-1, k) * pow(k-j, n-1) * pow(2, n - 2*k)
            b = scipy.special.binom(n-1, k) * scipy.special.binom(2*k, j)
            y -= pow(1j, n) * n * (c * s * p * b) / (k + 1)
    return np_real(y, out)

def arcsinh(n, x, out=None):
    if n == 0:
        return np.arcsinh(x, out)
    else:
        x1 = np_recip_sqrt(1 + np.square(x))
        a = pow(-1, n-1) * math.factorial(n-1) * pow(x1, n)
        b = scipy.special.eval_legendre(n-1, x * x1)
        return np.multiply(a, b, out)

def arccosh(n, x, out=None):
    if n == 0:
        return np.arccosh(x, out)
    else:
        x1 = np_recip_sqrt(1 - np.square(x))
        a = -1 * pow(-1j, n) * math.factorial(n-1) * pow(x1, n)
        b = scipy.special.eval_legendre(n-1, 1j * x * x1)
        return np_real(a*b, out)

def arctanh(n, x, out=None):
    if n == 0:
        return np.arctanh(x, out)
    else:
        out = np.add(pow(1 - x, -n), pow(-1, n-1) * pow(x+1, -n), out)
        out *= 0.5 * math.factorial(n-1)
        return out


##############################################################################
# Hypergeometric functions.

def hyp0f1(b, n, x, out=None):
    out = scipy.special.hyp0f1(b+n, x, out)
    out /= scipy.special.poch(b, n)
    return out

def hyp1f1(a, b, n, x, out=None):
    out = scipy.special.hyp1f1(a+n, b+n, x, out)
    out *= scipy.special.poch(a, n) / scipy.special.poch(b, n)
    return out

def hyperu(a, b, n, x, out=None):
    out = scipy.special.hyperu(a+n, b+n, x, out)
    out *= pow(-1, n) * scipy.special.poch(a, n)
    return out


##############################################################################
# Testing.
# FIXME: move the tests into a separate file.


class Test_DumbDeriv_Log(numpy.testing.TestCase):

    def test_log_n0(self):
        x = 0.123
        a = log(0, x)
        b = np.log(x)
        assert_allclose(a, b)

    def test_log_n1(self):
        x = 0.123
        a = log(1, x)
        b = 1 / x
        assert_allclose(a, b)

    def test_log_n2(self):
        x = 0.123
        a = log(2, x)
        b = -1 / np.square(x)
        assert_allclose(a, b)

    def test_log_n3(self):
        x = 0.123
        a = log(3, x)
        b = 2 / np.power(x, 3)
        assert_allclose(a, b)

    def test_log1p_n0(self):
        x = 0.123
        a = log1p(0, x)
        b = np.log1p(x)
        assert_allclose(a, b)

    def test_log1p_n1(self):
        x = 0.123
        a = log1p(1, x)
        b = 1 / (x + 1)
        assert_allclose(a, b)

    def test_log1p_n2(self):
        x = 0.123
        a = log1p(2, x)
        b = -1 / np.square(x + 1)
        assert_allclose(a, b)

    def test_log1p_n3(self):
        x = 0.123
        a = log1p(3, x)
        b = 2 / np.power(x + 1, 3)
        assert_allclose(a, b)



class Test_DumbDeriv_Misc(numpy.testing.TestCase):

    def test_reciprocal(self):
        x = 0.123
        a = reciprocal(0, x)
        b = np.reciprocal(x)
        assert_allclose(a, b)

    def test_cos(self):
        x = 0.123
        a = cos(0, x)
        b = np.cos(x)
        assert_allclose(a, b)

    def test_tan(self):
        x = 0.123
        a = tan(0, x)
        b = np.tan(x)
        assert_allclose(a, b)

    def test_arctan_n0(self):
        x = 0.123
        a = arctan(0, x)
        b = np.arctan(x)
        assert_allclose(a, b)

    def test_arctan_n1(self):
        x = 0.123
        a = arctan(1, x)
        b = 1 / (x*x + 1)
        assert_allclose(a, b)

    def test_cosh(self):
        x = 0.123
        a = cosh(0, x)
        b = np.cosh(x)
        assert_allclose(a, b)

    def test_sqrt_n0(self):
        x = 0.123
        a = sqrt(0, x)
        b = np.sqrt(x)
        assert_allclose(a, b)

    def test_sqrt_n1(self):
        x = 0.123
        a = sqrt(1, x)
        b = 1 / (2 * np.sqrt(x))
        assert_allclose(a, b)

    def test_arccosh_n0(self):
        x = 2.345
        a = arccosh(0, x)
        b = np.arccosh(x)
        assert_allclose(a, b)

    def test_arccosh_n1(self):
        x = 2.345
        a = arccosh(1, x)
        b = np_recip_sqrt(x*x - 1)
        assert_allclose(a, b)


if __name__ == '__main__':
    numpy.testing.run_module_suite()

