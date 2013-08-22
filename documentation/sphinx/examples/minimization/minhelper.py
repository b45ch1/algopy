"""
This is a helper module for the minimization examples.
"""

#FIXME: the pyipopt stuff has been commented out for now;
#FIXME: update this code to use a better python ipopt interface when available
# https://github.com/xuy/pyipopt
# http://gitview.danfis.cz/pipopt
# https://bitbucket.org/amitibo
# https://github.com/casadi/casadi

import functools

import numpy
import scipy.optimize
import algopy
import numdifftools
#import pyipopt

# Suppress log spam from pyipopt.
# But ipopt itself will stil spam...
#pyipopt.set_loglevel(0)


def eval_grad(f, theta):
    theta = algopy.UTPM.init_jacobian(theta)
    return algopy.UTPM.extract_jacobian(f(theta))

def eval_hess(f, theta):
    theta = algopy.UTPM.init_hessian(theta)
    return algopy.UTPM.extract_hessian(len(theta), f(theta))



def show_local_curvature(f, g, h, x0):
    print('point:')
    print(x0)
    print('function value:')
    print(f(x0))
    print('autodiff gradient:')
    print(g(x0))
    print('finite differences gradient:')
    print(numdifftools.Gradient(f)(x0))
    print('autodiff hessian:')
    print(h(x0))
    print('finite differences hessian:')
    print(numdifftools.Hessian(f)(x0))


def do_searches(f, g, h, x0):

    print('properties of the function at the initial guess:')
    show_local_curvature(f, g, h, x0)
    print()

    print('strategy:', 'default (Nelder-Mead)')
    print('options:', 'default')
    results = scipy.optimize.fmin(
            f,
            x0,
            )
    print(results)
    print()

    print('strategy:', 'ncg')
    print('options:', 'default')
    print('gradient:', 'autodiff')
    print('hessian:', 'autodiff')
    results = scipy.optimize.fmin_ncg(
            f,
            x0,
            fprime=g,
            fhess=h,
            )
    print(results)
    print()

    print('strategy:', 'ncg')
    print('options:', 'default')
    print('gradient:', 'autodiff')
    print('hessian:', 'finite differences')
    results = scipy.optimize.fmin_ncg(
            f,
            x0,
            fprime=g,
            )
    print(results)
    print()

    print('strategy:', 'cg')
    print('options:', 'default')
    print('gradient:', 'autodiff')
    results = scipy.optimize.fmin_cg(
            f,
            x0,
            fprime=g,
            )
    print(results)
    print()

    print('strategy:', 'cg')
    print('options:', 'default')
    print('gradient:', 'finite differences')
    results = scipy.optimize.fmin_cg(
            f,
            x0,
            )
    print(results)
    print()

    print('strategy:', 'bfgs')
    print('options:', 'default')
    print('gradient:', 'autodiff')
    results = scipy.optimize.fmin_bfgs(
            f,
            x0,
            fprime=g,
            )
    print(results)
    print()

    print('strategy:', 'bfgs')
    print('options:', 'default')
    print('gradient:', 'finite differences')
    results = scipy.optimize.fmin_bfgs(
            f,
            x0,
            )
    print(results)
    print()

    print('strategy:', 'slsqp')
    print('options:', 'default')
    print('gradient:', 'autodiff')
    results = scipy.optimize.fmin_slsqp(
            f,
            x0,
            fprime=g,
            )
    print(results)
    print()

    print('strategy:', 'slsqp')
    print('options:', 'default')
    print('gradient:', 'finite differences')
    results = scipy.optimize.fmin_slsqp(
            f,
            x0,
            )
    print(results)
    print()

    print('strategy:', 'powell')
    print('options:', 'default')
    results = scipy.optimize.fmin_powell(
            f,
            x0,
            )
    print(results)
    print()

    print('strategy:', 'tnc')
    print('options:', 'default')
    print('gradient:', 'autodiff')
    results = scipy.optimize.fmin_tnc(
            f,
            x0,
            fprime=g,
            disp=0,
            )
    print(results)
    print()

    print('strategy:', 'tnc')
    print('options:', 'default')
    print('gradient:', 'finite differences')
    results = scipy.optimize.fmin_tnc(
            f,
            x0,
            approx_grad=True,
            disp=0,
            )
    print(results)
    print()

    #print 'strategy:', 'ipopt'
    #print 'options:', 'default'
    #print 'gradient:', 'autodiff'
    #print 'hessian:', 'autodiff'
    #results = pyipopt.fmin_unconstrained(
            #f,
            #x0,
            #fprime=g,
            #fhess=h,
            #)
    #print results
    #print

    #print 'strategy:', 'ipopt'
    #print 'options:', 'default'
    #print 'gradient:', 'autodiff'
    #print 'hessian:', 'finite differences'
    #results = pyipopt.fmin_unconstrained(
            #f,
            #x0,
            #fprime=g,
            #)
    #print results
    #print


def show_minimization_results(f, target_in, easy_init_in, hard_init_in):
    """
    Print some results related to the minimization of the objective function.
    @param f: this is the objective function
    @param target_in: this is the min point
    @param easy_init_in: an easier starting point
    @param hard_init_in: a harder starting point
    """

    # the points are now float arrays
    target = numpy.array(target_in, dtype=float)
    easy_init = numpy.array(easy_init_in, dtype=float)
    hard_init = numpy.array(hard_init_in, dtype=float)

    # define the function and the autodiff gradient and hessian
    g = functools.partial(eval_grad, f)
    h = functools.partial(eval_hess, f)

    x0 = target
    print('properties of the function at a local min:')
    show_local_curvature(f, g, h, x0)
    print()

    x0 = easy_init
    print('---------------------------------------------------------')
    print('searches beginning from the easier init point', x0)
    print('---------------------------------------------------------')
    print()
    do_searches(f, g, h, x0)
    print()

    x0 = hard_init
    print('---------------------------------------------------------')
    print('searches beginning from the more difficult init point', x0)
    print('---------------------------------------------------------')
    print()
    do_searches(f, g, h, x0)
    print()

