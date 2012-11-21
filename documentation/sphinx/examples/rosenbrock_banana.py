"""
Minimize the Rosenbrock banana function.

http://en.wikipedia.org/wiki/Rosenbrock_function
"""

import functools

import numpy
import scipy.optimize
import algopy


#########################################################################
# These are helper functions for algopy.
# The argument order is compatible with functools.partial().
# Possibly they are overkill for this example
# because the extra args are not used.


def eval_grad(f, theta, *args):
    theta = algopy.UTPM.init_jacobian(theta)
    retval = f(theta, *args)
    return algopy.UTPM.extract_jacobian(retval)

def eval_hess(f, theta, *args):
    theta = algopy.UTPM.init_hessian(theta)
    retval = f(theta, *args)
    return algopy.UTPM.extract_hessian(len(theta), retval)


#########################################################################
# This is the single example-specific function to be minimized.


def rosenbrock(X):
    """
    This R^2 -> R^1 function should be compatible with algopy.
    http://en.wikipedia.org/wiki/Rosenbrock_function
    """
    x = X[0]
    y = X[1]
    a = 1. - x
    b = y - x*x
    return a*a + b*b*100.


#########################################################################
# Compare optimization strategies.


def main():

    # define the function and the autodiff gradient and hessian
    f = rosenbrock
    g = functools.partial(eval_grad, rosenbrock)
    h = functools.partial(eval_hess, rosenbrock)

    print 'Try to find the minimum of the Rosenbrock banana function.'
    print 'This is at f(1, 1) = 0 but the function is a bit tricky.'
    print 'To make the search difficult we will start far from the min.'
    print

    x0 = numpy.array([-1.2, 1.0], dtype=float)
    print 'initial guess:'
    print x0
    print

    print 'strategy:', 'default (Nelder-Mead)'
    print 'options:', 'default'
    results = scipy.optimize.fmin(
            f,
            x0,
            )
    print results
    print

    print 'strategy:', 'ncg'
    print 'options:', 'default'
    print 'gradient:', 'autodiff'
    print 'hessian:', 'autodiff'
    results = scipy.optimize.fmin_ncg(
            f,
            x0,
            fprime=g,
            fhess=h,
            )
    print results
    print

    print 'strategy:', 'ncg'
    print 'options:', 'default'
    print 'gradient:', 'autodiff'
    print 'hessian:', 'finite differences'
    results = scipy.optimize.fmin_ncg(
            f,
            x0,
            fprime=g,
            )
    print results
    print

    print 'strategy:', 'bfgs'
    print 'options:', 'default'
    print 'gradient:', 'autodiff'
    results = scipy.optimize.fmin_bfgs(
            f,
            x0,
            fprime=g,
            )
    print results
    print

    print 'strategy:', 'bfgs'
    print 'options:', 'default'
    print 'gradient:', 'finite differences'
    results = scipy.optimize.fmin_bfgs(
            f,
            x0,
            )
    print results
    print

    print 'strategy:', 'slsqp'
    print 'options:', 'default'
    print 'gradient:', 'autodiff'
    results = scipy.optimize.fmin_slsqp(
            f,
            x0,
            fprime=g,
            )
    print results
    print

    print 'strategy:', 'slsqp'
    print 'options:', 'default'
    print 'gradient:', 'finite differences'
    results = scipy.optimize.fmin_slsqp(
            f,
            x0,
            )
    print results
    print

    print 'strategy:', 'powell'
    print 'options:', 'default'
    results = scipy.optimize.fmin_powell(
            f,
            x0,
            )
    print results
    print


if __name__ == '__main__':
    main()

