"""
This is a helper module for the minimization examples.
"""

import functools

import numpy
import scipy.optimize
import algopy
import numdifftools
import pyipopt


#######################################################################
# FIXME:
# The variables and functions in this section are ipopt-specific
# and should be moved somewhere else.


# http://www.coin-or.org/Ipopt/documentation/node35.html
g_ipopt_pos_inf = 2e19
g_ipopt_neg_inf = -g_ipopt_pos_inf

def my_ipopt_dummy_eval_g(X, user_data=None):
    return numpy.array([], dtype=float)

def my_ipopt_dummy_eval_jac_g(X, flag, user_data=None):
    """
    We assume that the optimization is unconstrained.
    Therefore the constraint Jacobian is not so interesting,
    but we still need it to exist so that ipopt does not complain.
    """
    rows = numpy.array([], dtype=int)
    cols = numpy.array([], dtype=int)
    if flag:
        return (rows, cols)
    else:
        raise Exception('this should not be called')
        #return numpy.array([], dtype=float)

def my_ipopt_eval_h(
        h, nvar,
        X, lagrange, obj_factor, flag, user_data=None):
    """
    Evaluate the sparse hessian of the Lagrangian.
    The first group of parameters should be applied using functools.partial.
    The second group of parameters are passed from ipopt.
    @param h: a function to compute the hessian.
    @param nvar: the number of parameters
    @param X: parameter values
    @param lagrange: something about the constraints
    @param obj_factor: no clue what this is
    @param flag: this asks for the sparsity structure
    """

    # Get the nonzero (row, column) entries of a lower triangular matrix.
    # This is related to the fact that the Hessian is symmetric,
    # and that ipopt is designed to work with sparse matrices.
    row_list = []
    col_list = []
    for row in range(nvar):
        for col in range(row+1):
            row_list.append(row)
            col_list.append(col)
    rows = numpy.array(row_list, dtype=int)
    cols = numpy.array(col_list, dtype=int)

    if flag:
        return (rows, cols)
    else:
        if nvar != len(X):
            raise Exception('parameter count mismatch')
        if lagrange:
            raise Exception('only unconstrained is implemented for now...')
        values = numpy.zeros(len(rows), dtype=float)
        H = h(X)
        for i, (r, c) in enumerate(zip(rows, cols)):
            #FIXME: am I using obj_factor correctly?
            # I don't really know what it is...
            values[i] = H[r, c] * obj_factor
        return values

def my_ipopt_apply_new(X):
    #FIXME: I don't really know what this does, but ipopt wants it.
    return True

def my_fmin_ipopt(f, x0, fprime, fhess=None):
    nvar = len(x0)
    x_L = numpy.array([g_ipopt_neg_inf]*nvar, dtype=float)
    x_U = numpy.array([g_ipopt_pos_inf]*nvar, dtype=float)
    ncon = 0
    g_L = numpy.array([], dtype=float)
    g_U = numpy.array([], dtype=float)
    nnzj = 0
    nnzh = (nvar * (nvar + 1)) // 2

    # define the callbacks
    eval_f = f
    eval_grad_f = fprime
    eval_g = my_ipopt_dummy_eval_g
    eval_jac_g = my_ipopt_dummy_eval_jac_g
    if fhess:
        eval_h = functools.partial(my_ipopt_eval_h, fhess, nvar)
        apply_new = my_ipopt_apply_new

    # compute the results using ipopt
    nlp_args = [
            nvar,
            x_L,
            x_U,
            ncon,
            g_L,
            g_U,
            nnzj,
            nnzh,
            eval_f,
            eval_grad_f,
            eval_g,
            eval_jac_g,
            ]
    if fhess:
        nlp_args.extend([
            eval_h,
            apply_new,
            ])
    nlp = pyipopt.create(*nlp_args)
    results = nlp.solve(x0)
    nlp.close()
    return results


#######################################################################
# The following two functions are algopy-specific.


def eval_grad(f, theta):
    theta = algopy.UTPM.init_jacobian(theta)
    return algopy.UTPM.extract_jacobian(f(theta))

def eval_hess(f, theta):
    theta = algopy.UTPM.init_hessian(theta)
    return algopy.UTPM.extract_hessian(len(theta), f(theta))



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

    print 'strategy:', 'cg'
    print 'options:', 'default'
    print 'gradient:', 'autodiff'
    results = scipy.optimize.fmin_cg(
            f,
            x0,
            fprime=g,
            )
    print results
    print

    print 'strategy:', 'cg'
    print 'options:', 'default'
    print 'gradient:', 'finite differences'
    results = scipy.optimize.fmin_cg(
            f,
            x0,
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

    print 'strategy:', 'ipopt'
    print 'options:', 'default'
    print 'gradient:', 'autodiff'
    print 'hessian:', 'autodiff'
    results = my_fmin_ipopt(
            f,
            x0,
            fprime=g,
            fhess=h,
            )
    print results
    print

    print 'strategy:', 'ipopt'
    print 'options:', 'default'
    print 'gradient:', 'autodiff'
    print 'hessian:', 'finite differences'
    results = my_fmin_ipopt(
            f,
            x0,
            fprime=g,
            )
    print results
    print


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
    print 'properties of the function at a local min:'
    show_local_curvature(f, g, h, x0)
    print

    x0 = easy_init
    print '---------------------------------------------------------'
    print 'searches beginning from the easier init point', x0
    print '---------------------------------------------------------'
    print
    do_searches(f, g, h, x0)
    print

    x0 = hard_init
    print '---------------------------------------------------------'
    print 'searches beginning from the more difficult init point', x0
    print '---------------------------------------------------------'
    print
    do_searches(f, g, h, x0)
    print

