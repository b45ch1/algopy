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
#
# FIXME: none of these functions are currently used


def denom_not_genic(c, d):
    c2d = c / (2.*d)
    asym_part = algopy.exp(-c)
    sym_a = 1. / (2.*d)
    sym_b = algopy.exp(-c2d*(d*d + 1.))
    hyper_a = (1. + d) * algopy.special.hyp1f1(0.5, 1.5, c2d*(1+d)**2)
    hyper_b = (1. - d) * algopy.special.hyp1f1(0.5, 1.5, c2d*(1-d)**2)
    sym_part = sym_a * sym_b * (hyper_a - hyper_b)
    return asym_part * sym_part

def denom_near_genic(c, d):
    """
    This function is better when both |d|<<1 and |d/c|<<1.
    """
    a0 = 1. / (2.*c)
    b01 = 1. / (1.+d)
    b02 = algopy.special.hyp2f0(1.0, 0.5, (2.*d)/(c*(1.+d)**2))
    b11 = algopy.exp(-2.*c) / (1.-d)
    b12 = algopy.special.hyp2f0(1.0, 0.5, (2.*d)/(c*(1.-d)**2))
    return a0 * (b01 * b02 - b11 * b12)

def denom_genic_a(c):
    return algopy.special.hyp1f1(1., 2., -2.*c)

def denom_genic_b(c):
    return (1 - algopy.exp(-2*c)) / (2*c)

def denom_neutral():
    return 1.

def denom_piecewise(c, d):
    """
    This glues together the analytical solution.
    It seems to be usually the case that either denom_near_genic
    or denom_not_genic will give a good solution to the integral,
    but I have not yet found a good criterion for switching between them.
    """
    if c == 0:
        return denom_neutral()
    elif d == 0:
        return denom_genic_a(c)
    elif d**2 < 0.05**2:
    #elif d**2 + (d/c)**2 < 1e-3:
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
    try:
        return 1. / denom_quad(c, d)
    except Exception as e:
        print c
        print d
        raise e

def transform_params(Y):
    mu = algopy.exp(Y[0])
    d = Y[1]
    return mu, d

def create_transition_matrix(mu, d, v):
    """
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
    try:
        D = d * numpy.sign(S)
    except Exception as e:
        print v
        print d
        print S
        raise e
    pre_Q = numpy.vectorize(numeric_fixation)(0.5*S, D)
    #pre_Q = numeric_fixation(0.5*S, D)
    #Q = mu * numpy.vectorize(numeric_fixation)(0.5*S, d*numpy.sign(S))
    #Q = mu * numpy.vectorize(numeric_fixation)(0.5*S, d)
    pre_Q = mu * pre_Q
    Q = pre_Q - algopy.diag(algopy.sum(pre_Q, axis=1))
    P = algopy.expm(Q)
    #print 'P:'
    #print P
    #print
    return P

def eval_f(subs_counts, v, Y):
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
    try:
        P = create_transition_matrix(mu, d, v)
    except Exception as e:
        print 'error creating the transition matrix'
        print Y
        print subs_counts
        print v
        raise e
    # define the scoring matrix
    vdiag = algopy.diag(v)
    J = algopy.dot(vdiag, P)
    S = algopy.log(J)
    #print 'scoring matrix (should be symmetric):'
    #print S
    #print
    return -algopy.sum(S * subs_counts)

#FIXME: unused
def eval_grad_f(Y):
    """
    compute the gradient of f in the forward mode of AD
    """
    Y = algopy.UTPM.init_jacobian(Y)
    retval = eval_f(Y)
    return algopy.UTPM.extract_jacobian(retval)

#FIXME: unused
def eval_hess_f(Y):
    """
    compute the hessian of f in the forward mode of AD
    """
    Y = algopy.UTPM.init_hessian(Y)
    retval = eval_f(Y)
    return algopy.UTPM.extract_hessian(len(Y), retval)


#FIXME: this is not enabled because the analytic solution is too poor
class Test_foo(TestCase):

    def test_analytic_integration_solution(self):
        for c in numpy.linspace(-3, 3, 11):
            for d in numpy.linspace(-0.05, 0.05, 21):
                x = denom_piecewise(c, d)
                y = denom_quad(c, d)
                z = d**2 + (d/c)**2
                print 'c:         ', c
                print 'd:         ', d
                print 'quad:      ', y
                print 'piecewise: ', x
                print 'method:    ', z
                print denom_not_genic(c, d)
                print denom_near_genic(c, d)
                if abs(y - x) / y < 1e-6:
                    print 'ok'
                else:
                    print '*** bad ***'
                print
        raise Exception

def main():

    # There are four states, each corresponding to a nucleotide.
    n = 4

    # Initialize some arbitrary parameter values for a simulation study.
    mu = 0.1
    d = -1.0
    v = numpy.array([0.12, 0.18, 0.3, 0.4])

    # Define the conditional transition probabilities
    # and the distribution over substitutions.
    log_mu = numpy.log(mu)
    P = create_transition_matrix(mu, d, v)
    J = numpy.dot(numpy.diag(v), P)

    # Sample some substitution counts.
    subs_counts = numpy.random.multinomial(
            100000, J.reshape(n*n)).reshape((n,n))

    # Get an empirical equilibrium distribution.
    v_emp_weights = numpy.zeros(n, dtype=float)
    v_emp_weights += numpy.sum(subs_counts, axis=0)
    v_emp_weights += numpy.sum(subs_counts, axis=1)
    v_emp = v_emp_weights / numpy.sum(v_emp_weights)

    #print J
    #print subs_counts
    #print v
    #print v_emp

    #Y = numpy.array([log_mu, d], dtype=float)
    #print eval_f(Y, subs_counts, v_emp)
    #print eval_f(Y, subs_counts, v)

    eval_f_curried = functools.partial(eval_f, subs_counts, v_emp)

    # initialize the guess with the actual simulation parameter values
    y_guess = numpy.array([log_mu, d], dtype=float)
    #print eval_f_curried(y_guess)
    result = scipy.optimize.fmin(
            eval_f_curried,
            y_guess,
            maxiter=10000,
            maxfun=10000,
            full_output=True,
            )
    """
    result = scipy.optimize.fmin_bfgs(
            eval_f_curried,
            y_guess,
            gtol=1e-5,
            #gtol=1e-8,
            maxiter=10000,
            full_output=True,
            )
    """
    print 'fmin result:'
    print result
    print
    y_opt = result[0]
    log_mu_opt, d_opt = y_opt[0], y_opt[1]
    mu_opt = numpy.exp(log_mu_opt)
    cov = scipy.linalg.inv(numdifftools.Hessian(eval_f_curried)(y_opt))
    print 'simulation parameter values:'
    print 'log mu:', log_mu
    print 'd:', d
    print
    print 'maximum likelihood parameter estimates:'
    print 'log mu:', log_mu_opt,
    print 'with standard deviation', numpy.sqrt(cov[0,0])
    print 'd:', d_opt,
    print 'with standard deviation', numpy.sqrt(cov[1,1])
    print


if __name__ == '__main__':
    #run_module_suite()
    main()

