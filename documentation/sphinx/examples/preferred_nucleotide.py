"""
Differentiation of a another function inspired by molecular biology.
"""

import functools

import numpy
from numpy.testing import *
from numpy.testing.decorators import *
import scipy
import scipy.integrate
import scipy.optimize
import numdifftools

import algopy
import algopy.special


###########################################################################
# These functions are for the analytical solution of a definite integral.


def denom_not_genic(c, d):
    c2d = c / (2.*d)
    asym_part = algopy.exp(-c)
    sym_a = 1. / (2.*d)
    sym_b = algopy.exp(-c2d*(d*d + 1.))
    hyper_a = (1. + d) * algopy.special.dpm_hyp1f1(0.5, 1.5, c2d*(1+d)**2)
    hyper_b = (1. - d) * algopy.special.dpm_hyp1f1(0.5, 1.5, c2d*(1-d)**2)
    sym_part = sym_a * sym_b * (hyper_a - hyper_b)
    return asym_part * sym_part

def denom_near_genic(c, d):
    a0 = 1. / (2.*c)
    b01 = 1. / (1.+d)
    b02 = algopy.special.dpm_hyp2f0(1.0, 0.5, (2.*d)/(c*(1.+d)**2))
    b11 = algopy.exp(-2.*c) / (1.-d)
    b12 = algopy.special.dpm_hyp2f0(1.0, 0.5, (2.*d)/(c*(1.-d)**2))
    return a0 * (b01 * b02 - b11 * b12)

def denom_genic(c):
    return algopy.special.dpm_hyp1f1(1., 2., -2.*c)

def denom_neutral():
    return 1.

def denom_piecewise(c, d):
    """
    This glues together the analytical solution.
    This is a second attempt, and this time it is
    with respect to the mpmath hypergeometric function implementations.
    """
    small_eps = 1e-8
    large_eps = 1e-3
    if abs(c) < small_eps:
        #FIXME: this does not give sufficient Taylor information about c or d
        return denom_neutral()
    elif abs(d) < small_eps:
        #FIXME: this does not give sufficient Taylor information about d
        return denom_genic(c)
    elif abs(d) > 1 - large_eps:
        return denom_not_genic(c, d)
    elif -1 < d/c < large_eps:
        return denom_near_genic(c, d)
    else:
        return denom_not_genic(c, d)


###########################################################################
# These functions are for the numerical solution of a definite integral.


def denom_integrand(x, c, d):
    return algopy.exp(-2*c*d*x*(1-x) - 2*c*x)

def denom_quad(c, d):
    result = scipy.integrate.quad(
            denom_integrand,
            0., 1.,
            args=(c,d),
            full_output=1,
            )
    return result[0]


###########################################################################
# Construct a stationary distribution and transition matrix.

def numeric_fixation(c, d):
    """
    This is related to the probability of a mutation fixing in a population.
    @param c: positive when preferred
    @param d: negative when recessive
    """
    return 1. / denom_quad(c, d)

def transform_params(Y):
    mu = algopy.exp(Y[0])
    d = Y[1]
    return mu, d

def create_transition_matrix_explicit(Y, v):
    """
    Use hypergeometric functions.
    Note that d = 2*h - 1 following Kimura 1957.
    The rate mu is a catch-all scaling factor.
    The finite distribution v is assumed to be a stochastic vector.
    @param Y: vector of parameters to optimize
    @param v: numpy array defining a distribution over states
    @return: transition matrix
    """

    n = len(v)
    mu, d = transform_params(Y)

    # Construct the numpy matrix whose entries
    # are differences of log equilibrium probabilities.
    # Everything in this code block is pure numpy.
    F = numpy.log(v)
    e = numpy.ones_like(F)
    S = numpy.outer(e, F) - numpy.outer(F, e)

    # Create the rate matrix Q and return its matrix exponential.
    # Things in this code block may use algopy if mu and d
    # are bundled with truncated Taylor information.
    D = d * numpy.sign(S)

    #FIXME: I would like to further vectorize this block,
    # and also it may currently give subtly wrong results
    # because denom_piecewise may not vectorize correctly.
    pre_Q = algopy.zeros((n,n), dtype=Y)
    for i in range(n):
        for j in range(n):
            pre_Q[i, j] = 1. / denom_piecewise(0.5*S[i, j], D[i, j])

    pre_Q = mu * pre_Q
    Q = pre_Q - algopy.diag(algopy.sum(pre_Q, axis=1))
    P = algopy.expm(Q)
    return P



def create_transition_matrix_numeric(mu, d, v):
    """
    Use numerical integration.
    This is not so compatible with algopy because it goes through fortran.
    Note that d = 2*h - 1 following Kimura 1957.
    The rate mu is a catch-all scaling factor.
    The finite distribution v is assumed to be a stochastic vector.
    @param mu: scales the rate matrix
    @param d: dominance (as opposed to recessiveness) of preferred states.
    @param v: numpy array defining a distribution over states
    @return: transition matrix
    """

    # Construct the numpy matrix whose entries
    # are differences of log equilibrium probabilities.
    # Everything in this code block is pure numpy.
    F = numpy.log(v)
    e = numpy.ones_like(F)
    S = numpy.outer(e, F) - numpy.outer(F, e)

    # Create the rate matrix Q and return its matrix exponential.
    # Things in this code block may use algopy if mu and d
    # are bundled with truncated Taylor information.
    D = d * numpy.sign(S)
    pre_Q = numpy.vectorize(numeric_fixation)(0.5*S, D)
    pre_Q = mu * pre_Q
    Q = pre_Q - algopy.diag(algopy.sum(pre_Q, axis=1))
    P = algopy.expm(Q)
    return P

def eval_f_explicit(subs_counts, v, Y):
    """
    Note that Y is last for compatibility with functools.partial.
    It is convenient for usage with numdifftools, although this parameter
    ordering is the opposite of the convention of scipy.optimize.
    @return: negative log likelihood
    @param Y: parameters to jointly estimate
    @param subs_counts: observed data
    @param v: fixed equilibrium probabilities for states
    """
    P = create_transition_matrix_explicit(Y, v)
    vdiag = algopy.diag(v)
    J = algopy.dot(vdiag, P)
    S = algopy.log(J)
    return -algopy.sum(S * subs_counts)

def eval_f_numeric(subs_counts, v, Y):
    """
    Note that Y is last for compatibility with functools.partial.
    It is convenient for usage with numdifftools, although this parameter
    ordering is the opposite of the convention of scipy.optimize.
    @return: negative log likelihood
    @param Y: parameters to jointly estimate
    @param subs_counts: observed data
    @param v: fixed equilibrium probabilities for states
    """
    mu, d = transform_params(Y)
    P = create_transition_matrix_numeric(mu, d, v)
    vdiag = algopy.diag(v)
    J = algopy.dot(vdiag, P)
    S = algopy.log(J)
    return -algopy.sum(S * subs_counts)

def eval_grad_f(subs_counts, v, Y):
    """
    compute the gradient of f in the forward mode of AD
    """
    Y = algopy.UTPM.init_jacobian(Y)
    retval = eval_f_explicit(subs_counts, v, Y)
    return algopy.UTPM.extract_jacobian(retval)

def eval_hess_f(subs_counts, v, Y):
    """
    compute the hessian of f in the forward mode of AD
    """
    Y = algopy.UTPM.init_hessian(Y)
    retval = eval_f_explicit(subs_counts, v, Y)
    return algopy.UTPM.extract_hessian(len(Y), retval)


def simulation_demo_numeric(log_mu, d, subs_counts, v_emp):
    eval_f_partial = functools.partial(eval_f_numeric, subs_counts, v_emp)
    # initialize the guess with the actual simulation parameter values
    y_guess = numpy.array([log_mu, d], dtype=float)
    result = scipy.optimize.fmin(
            eval_f_partial,
            y_guess,
            maxiter=10000,
            maxfun=10000,
            full_output=True,
            )
    print('fmin result:')
    print(result)
    print()
    y_opt = result[0]
    log_mu_opt, d_opt = y_opt[0], y_opt[1]
    mu_opt = numpy.exp(log_mu_opt)
    cov = scipy.linalg.inv(numdifftools.Hessian(eval_f_partial)(y_opt))
    print('maximum likelihood parameter estimates:')
    print('log mu:', log_mu_opt, end=' ')
    print('with standard deviation', numpy.sqrt(cov[0,0]))
    print('d:', d_opt, end=' ')
    print('with standard deviation', numpy.sqrt(cov[1,1]))
    print()
    print('gradient:')
    print(numdifftools.Gradient(eval_f_partial)(y_opt))
    print()


def simulation_demo_explicit(log_mu, d, subs_counts, v_emp):
    eval_f_partial = functools.partial(eval_f_explicit, subs_counts, v_emp)
    eval_grad_f_partial = functools.partial(eval_grad_f, subs_counts, v_emp)
    eval_hess_f_partial = functools.partial(eval_hess_f, subs_counts, v_emp)
    # initialize the guess with the actual simulation parameter values
    y_guess = numpy.array([log_mu, d], dtype=float)
    result = scipy.optimize.fmin_ncg(
            eval_f_partial,
            y_guess,
            fprime=eval_grad_f_partial,
            fhess=eval_hess_f_partial,
            maxiter=10000,
            full_output=True,
            )
    print('fmin_ncg result:')
    print(result)
    print()
    y_opt = result[0]
    log_mu_opt, d_opt = y_opt[0], y_opt[1]
    mu_opt = numpy.exp(log_mu_opt)
    cov = scipy.linalg.inv(eval_hess_f_partial(y_opt))
    print('maximum likelihood parameter estimates:')
    print('log mu:', log_mu_opt, end=' ')
    print('with standard deviation', numpy.sqrt(cov[0,0]))
    print('d:', d_opt, end=' ')
    print('with standard deviation', numpy.sqrt(cov[1,1]))
    print()
    print('gradient:')
    print(eval_grad_f_partial(y_opt))
    print()


def main():

    # There are four states, each corresponding to a nucleotide.
    n = 4

    # Initialize some arbitrary parameter values for a simulation study.
    mu = 0.1

    # Initialize arbitrary equilibrium nucleotide distribution.
    v = numpy.array([0.12, 0.18, 0.3, 0.4])

    # Sample this many substitutions.
    nsamples = 100000

    for d in (-0.01, 0.01, 2.2):

        # Define the conditional transition probabilities
        # and the distribution over substitutions.
        log_mu = numpy.log(mu)
        P = create_transition_matrix_numeric(mu, d, v)
        J = numpy.dot(numpy.diag(v), P)

        # Sample some substitution counts.
        subs_counts = numpy.random.multinomial(
                nsamples, J.reshape(n*n)).reshape((n,n))

        # Get an empirical equilibrium distribution.
        v_emp_weights = numpy.zeros(n, dtype=float)
        v_emp_weights += numpy.sum(subs_counts, axis=0)
        v_emp_weights += numpy.sum(subs_counts, axis=1)
        v_emp = v_emp_weights / numpy.sum(v_emp_weights)

        print('simulation parameter values:')
        print('log mu:', log_mu)
        print('d:', d)
        print('nsamples:', nsamples)
        print('sampled substitutions:')
        print(subs_counts)
        print()
        print('===============================================================')
        print('--- estimation via numerical integration, numdifftools, fmin --')
        simulation_demo_numeric(log_mu, d, subs_counts, v_emp)
        print('---------------------------------------------------------------')
        print()
        print('===============================================================')
        print('--- estimation via hypergeometric functions, algopy, fmin_ncg -')
        simulation_demo_explicit(log_mu, d, subs_counts, v_emp)
        print('---------------------------------------------------------------')
        print()


if __name__ == '__main__':
    main()

