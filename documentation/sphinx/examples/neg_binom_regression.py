"""
Negative Binomial Regression.

http://statsmodels.sourceforge.net/devel/examples/generated/example_gmle.html
"""

import functools

import numpy
import scipy.optimize
import algopy
import numdifftools
import pandas
import patsy

g_url = 'http://vincentarelbundock.github.com/Rdatasets/csv/COUNT/medpar.csv'

def get_aic(theta, y, X):
    return 2*len(theta) + 2*get_neg_ll(theta, y, X)

def get_neg_ll(theta, y, X):
    alpha = theta[-1]
    beta = theta[:-1]
    a = alpha * algopy.exp(algopy.dot(X, beta))
    ll = algopy.sum(
        -y*algopy.log1p(1/a) +
        -algopy.log1p(a) / alpha +
        algopy.special.gammaln(y + 1/alpha) +
        -algopy.special.gammaln(y + 1) +
        -algopy.special.gammaln(1/alpha))
    neg_ll = -ll
    #print theta
    #print neg_ll
    #print
    return neg_ll

def eval_grad(f, theta, *args):
    theta = algopy.UTPM.init_jacobian(theta)
    retval = f(theta, *args)
    return algopy.UTPM.extract_jacobian(retval)

def eval_hess(f, theta, *args):
    theta = algopy.UTPM.init_hessian(theta)
    retval = f(theta, *args)
    return algopy.UTPM.extract_hessian(len(theta), retval)


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

    # read the data into numpy arrays
    medpar = pandas.read_csv(g_url)
    y, X = patsy.dmatrices('los~type2+type3+hmo+white', medpar)
    y = numpy.array(y).flatten()
    X = numpy.array(X)
    print type(y)
    print type(X)
    print y[:5]
    print X[:5]

    fmin_args = (y, X)

    # define the function and the autodiff gradient and hessian
    f = get_neg_ll
    g = functools.partial(eval_grad, get_neg_ll)
    h = functools.partial(eval_hess, get_neg_ll)

    # define the max likelihood values from the statsmodels webpage
    expected_theta = numpy.array([
        2.3103, 0.2213, 0.7061, -0.068, -0.129, 0.4458])
    expected_aic = 9606.9532058301575
    print 'expected aic:'
    print expected_aic
    print
    print 'observed aic:'
    print get_aic(expected_theta, y, X)
    print
    print 'standard deviations at expected mle:'
    print numpy.sqrt(numpy.diag(scipy.linalg.inv(h(expected_theta, y, X))))
    print

    # init the search for max likelihood parameters
    theta0 = numpy.array([
        numpy.log(numpy.mean(y)),
        0, 0, 0, 0,
        0.5,
        ], dtype=float)

    # do the max likelihood search
    results = scipy.optimize.fmin_ncg(
            f,
            theta0,
            fprime=g,
            fhess=h,
            args=fmin_args,
            avextol=1e-6,
            )

    print 'search results:'
    print results
    print
    print 'standard deviations at observed mle:'
    print numpy.sqrt(numpy.diag(scipy.linalg.inv(h(results, y, X))))
    print


if __name__ == '__main__':
    main()

