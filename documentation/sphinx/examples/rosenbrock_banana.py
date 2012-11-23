"""
Minimize the Rosenbrock banana function.

http://en.wikipedia.org/wiki/Rosenbrock_function
"""

import functools

import numpy
import scipy.optimize
import algopy
import numdifftools


#########################################################################
# All of the example-specific stuff is in this section.
# FIXME: Everything else is copypasted and should possibly be reorganized.
# See also the himmelblau_minimization.py example.


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

g_objective_function = rosenbrock
g_target = numpy.array([1, 1], dtype=float)
g_easy_init = numpy.array([2, 2], dtype=float)
g_hard_init = numpy.array([-1.2, 1], dtype=float)


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
# Compare optimization strategies.

def do_searches(f, g, h, x0):

    print 'properties of the function at the initial guess:'
    show_local_curvature(f, g, h, x0)
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

    print 'strategy:', 'tnc'
    print 'options:', 'default'
    print 'gradient:', 'autodiff'
    results = scipy.optimize.fmin_tnc(
            f,
            x0,
            fprime=g,
            disp=0,
            )
    print results
    print

    print 'strategy:', 'tnc'
    print 'options:', 'default'
    print 'gradient:', 'finite differences'
    results = scipy.optimize.fmin_tnc(
            f,
            x0,
            approx_grad=True,
            disp=0,
            )
    print results
    print


def show_local_curvature(f, g, h, x0):
    print 'point:'
    print x0
    print 'function value:'
    print f(x0)
    print 'autodiff gradient:'
    print g(x0)
    print 'finite differences gradient:'
    print numdifftools.Gradient(f)(x0)
    print 'autodiff hessian:'
    print h(x0)
    print 'finite differences hessian:'
    print numdifftools.Hessian(f)(x0)


def main():

    # define the function and the autodiff gradient and hessian
    f = g_objective_function
    g = functools.partial(eval_grad, g_objective_function)
    h = functools.partial(eval_hess, g_objective_function)

    x0 = g_target
    print 'properties of the function at a local min:'
    show_local_curvature(f, g, h, x0)
    print

    x0 = g_easy_init
    print '---------------------------------------------------------'
    print 'searches beginning from the easier init point', x0
    print '---------------------------------------------------------'
    print
    do_searches(f, g, h, x0)
    print

    x0 = g_hard_init
    print '---------------------------------------------------------'
    print 'searches beginning from the more difficult init point', x0
    print '---------------------------------------------------------'
    print
    do_searches(f, g, h, x0)
    print


if __name__ == '__main__':
    main()

