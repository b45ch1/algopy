"""
Univariate nth derivatives of several numpy and scipy functions.

These functions are intended for two purposes.
They can be used for testing,
and they can also be used as components of
unsophisticated implementations of more complicated functions.
The functions in this module do not support complex numbers
as input or output types, although some of the functions use them internally.
Sometimes scipy is used for sqrt because it can deal with negative numbers.
Tests are in a separate module.
"""


import functools
import math

import numpy as np
import numpy.testing
from numpy.testing import assert_allclose

# fail later in the process, by raising import warnings instead of errors
try:
    import scipy
    try:
        import scipy.special
    except ImportError as e:
        raise ImportWarning(e)
except ImportError as e:
    raise ImportWarning(e)

# Define some function domain constants for usage in testing.
DOM_ALL = 'all real numbers'
DOM_POS = 'positive real numbers'
DOM_GT_1 = 'real numbers greater than 1'
DOM_GT_NEG_1 = 'real numbers greater than -1'
DOM_ABS_LT_1 = 'real numbers with absolute value less than 1'

# Enumerate the domain constants for error checking.
_domain_constants = (
        DOM_ALL,
        DOM_POS,
        DOM_GT_1,
        DOM_GT_NEG_1,
        DOM_ABS_LT_1,
        )

# Define the names of functions and constants to export.
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
        'DOM_ALL', 'DOM_POS', 'DOM_GT_1', 'DOM_GT_NEG_1', 'DOM_ABS_LT_1',
        ]


##############################################################################
# Define a decorator.
# Python 3 would allow nicer function definitions.
# http://stackoverflow.com/questions/5940180

def basecase(fn_zeroth_deriv, domain=DOM_ALL):
    """
    Deal with zeroth order derivatives.
    Make some effort to describe the domain of the function.
    """
    if domain not in _domain_constants:
        raise ValueError
    def wrap(f):
        def wrapped_f(*args, **kwargs):
            out = kwargs.pop('out', None)
            n = kwargs.pop('n', 0)
            if kwargs:
                raise ValueError('unexpected keyword args: %s' % kwargs)
            if n:
                return f(*args, out=out, n=n)
            elif out is None:
                return fn_zeroth_deriv(*args)
            else:
                return fn_zeroth_deriv(*args, out=out)
        wrapped_f.__name__ = fn_zeroth_deriv.__name__
        wrapped_f.__doc__ = fn_zeroth_deriv.__doc__
        wrapped_f.domain = domain
        return wrapped_f
    return wrap


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

@basecase(np.rint)
def rint(x, out=None, n=0):
    return np_filled_like(x, 0, out)

@basecase(np.fix)
def fix(x, out=None, n=0):
    return np_filled_like(x, 0, out)

@basecase(np.floor)
def floor(x, out=None, n=0):
    return np_filled_like(x, 0, out)

@basecase(np.ceil)
def ceil(x, out=None, n=0):
    return np_filled_like(x, 0, out)

@basecase(np.trunc)
def trunc(x, out=None, n=0):
    return np_filled_like(x, 0, out)

@basecase(np.sign)
def sign(x, out=None, n=0):
    return np_filled_like(x, 0, out)

@basecase(np.absolute)
def absolute(x, out=None, n=0):
    if n == 1:
        return np.sign(x, out)
    else:
        return np_filled_like(x, 0, out)


##############################################################################
# Miscellaneous special functions.

@basecase(scipy.special.erf)
def erf(x, out=None, n=0):
    a = pow(2, -n) * math.factorial(n - 1) * np.reciprocal(np_sqrt_pi)
    b = np.exp(-np.square(x))
    c = np.zeros_like(x)
    for k in range(1, n+1):
        s_a = pow(-1, k-1) * pow(2, 2*k) * pow(x, 2*k - n - 1)
        s_b = math.factorial(2*k - n - 1) * math.factorial(n - k)
        c += s_a * s_b
    return np.multiply(a*b, c, out)

@basecase(scipy.special.erfi)
def erfi(x, out=None, n=0):
    a = pow(2, -n) * math.factorial(n - 1) * np.reciprocal(np_sqrt_pi)
    b = np.exp(np.square(x))
    c = np.zeros_like(x)
    for k in range(1, n+1):
        s_a = pow(2, 2*k) * pow(x, 2*k - n - 1)
        s_b = math.factorial(2*k - n - 1) * math.factorial(n - k)
        c += s_a * s_b
    return np.multiply(a*b, c, out)

@basecase(scipy.special.gammaln, DOM_POS)
def gammaln(x, out=None, n=0):
    return scipy.special.polygamma(n-1, x, out)

@basecase(scipy.special.psi)
def psi(x, out=None, n=0):
    return scipy.special.polygamma(n, x, out)

@basecase(scipy.special.polygamma)
def polygamma(m, x, out=None, n=0):
    return scipy.special.polygamma(m+n, x, out)


##############################################################################
# Exponential, log, power, and polynomial functions.

@basecase(np.exp)
def exp(x, out=None, n=0):
    return np.exp(x, out)

@basecase(np.exp2)
def exp2(x, out=None, n=0):
    out = np.exp2(x, out)
    out *= pow(np.log(2), n)
    return out

@basecase(np.expm1)
def expm1(x, out=None, n=0):
    return np.exp(x, out)

@basecase(np.log, DOM_POS)
def log(x, out=None, n=0):
    out = np.power(x, -n, out)
    out *= pow(-1, n-1) * math.factorial(n-1)
    return out

@basecase(np.log2, DOM_POS)
def log2(x, out=None, n=0):
    out = log(x, out, n)
    out /= np.log(2)
    return out

@basecase(np.log10, DOM_POS)
def log10(x, out=None, n=0):
    out = log(x, out, n)
    out /= np.log(10)
    return out

@basecase(np.log1p, DOM_GT_NEG_1)
def log1p(x, out=None, n=0):
    out = np.power(1 + x, -n, out)
    out *= pow(-1, n-1) * math.factorial(n-1)
    return out

@basecase(np.sqrt, DOM_POS)
def sqrt(x, out=None, n=0):
    out = np.power(x, 0.5 - n, out)
    out *= scipy.special.poch(1.5 - n, n)
    return out

@basecase(np.square)
def square(x, out=None, n=0):
    if n == 1:
        return np.multiply(x, 2, out)
    elif n == 2:
        return np_filled_like(x, 2, out)
    else:
        return np_filled_like(x, 0, out)

@basecase(np.negative)
def negative(x, out=None, n=0):
    if n == 1:
        return np_filled_like(x, -1, out)
    else:
        return np_filled_like(x, 0, out)

@basecase(np.reciprocal)
def reciprocal(x, out=None, n=0):
    out = np.power(x, -(n+1), out)
    out *= math.factorial(n) * pow(-1, n)
    return out


##############################################################################
# Trigonometric functions and their functional inverses.

@basecase(np.sin)
def sin(x, out=None, n=0):
    return np.sin(0.5 * n * np.pi + x, out)

@basecase(np.cos)
def cos(x, out=None, n=0):
    return np.cos(0.5 * n * np.pi + x, out)

@basecase(np.tan)
def tan(x, out=None, n=0):
    out = np_filled_like(x, 0, out)
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

@basecase(np.arcsin, DOM_ABS_LT_1)
def arcsin(x, out=None, n=0):
    x1 = np_recip_sqrt(1 - np.square(x))
    a = 1j * pow(-1j, n) * math.factorial(n-1) * pow(x1, n)
    b = scipy.special.eval_legendre(n-1, 1j * x * x1)
    return np_real(a*b, out)

@basecase(np.arccos, DOM_ABS_LT_1)
def arccos(x, out=None, n=0):
    return np.negative(arcsin(x), out)

@basecase(np.arctan)
def arctan(x, out=None, n=0):
    a = 0.5j * pow(-1, n) * math.factorial(n - 1)
    b = pow(x - 1j, -n) - pow(x + 1j, -n)
    return np_real(a*b, out)


##############################################################################
# Hyperbolic trigonometric functions and their functional inverses.

@basecase(np.sinh)
def sinh(x, out=None, n=0):
    return np_real(pow(-1j, n) * np.sinh(0.5j * np.pi * n + x), out)

@basecase(np.cosh)
def cosh(x, out=None, n=0):
    return np_real(pow(-1j, n) * np.cosh(0.5j * np.pi * n + x), out)

@basecase(np.tanh)
def tanh(x, out=None, n=0):
    y = np.zeros_like(x, dtype=complex)
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

@basecase(np.arcsinh)
def arcsinh(x, out=None, n=0):
    x1 = np_recip_sqrt(1 + np.square(x))
    a = pow(-1, n-1) * math.factorial(n-1) * pow(x1, n)
    b = scipy.special.eval_legendre(n-1, x * x1)
    return np.multiply(a, b, out)

@basecase(np.arccosh, DOM_GT_1)
def arccosh(x, out=None, n=0):
    x1 = np_recip_sqrt(1 - np.square(x))
    a = -1 * pow(-1j, n) * math.factorial(n-1) * pow(x1, n)
    b = scipy.special.eval_legendre(n-1, 1j * x * x1)
    return np_real(a*b, out)

@basecase(np.arctanh, DOM_ABS_LT_1)
def arctanh(x, out=None, n=0):
    out = np.add(pow(1 - x, -n), pow(-1, n-1) * pow(x+1, -n), out)
    out *= 0.5 * math.factorial(n-1)
    return out


##############################################################################
# Hypergeometric functions.

@basecase(scipy.special.hyp0f1)
def hyp0f1(b, x, out=None, n=0):
    out = scipy.special.hyp0f1(b+n, x, out)
    out /= scipy.special.poch(b, n)
    return out

@basecase(scipy.special.hyp1f1)
def hyp1f1(a, b, x, out=None, n=0):
    out = scipy.special.hyp1f1(a+n, b+n, x, out)
    out *= scipy.special.poch(a, n) / scipy.special.poch(b, n)
    return out

@basecase(scipy.special.hyperu)
def hyperu(a, b, x, out=None, n=0):
    out = scipy.special.hyperu(a+n, b+n, x, out)
    out *= pow(-1, n) * scipy.special.poch(a, n)
    return out
