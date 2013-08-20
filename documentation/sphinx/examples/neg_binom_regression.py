"""
Negative Binomial Regression

This is an algopy implementation of the statsmodels example
http://statsmodels.sourceforge.net/devel/examples/generated/example_gmle.html
.
"""

import functools

import numpy
import scipy.optimize
import algopy
import numdifftools
import pandas
import patsy

g_url = 'http://vincentarelbundock.github.com/Rdatasets/csv/COUNT/medpar.csv'

def get_aic(y, X, theta):
    return 2*len(theta) + 2*get_neg_ll(y, X, theta)

def get_neg_ll(y, X, theta):
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
    return neg_ll

def eval_grad(f, theta):
    theta = algopy.UTPM.init_jacobian(theta)
    return algopy.UTPM.extract_jacobian(f(theta))

def eval_hess(f, theta):
    theta = algopy.UTPM.init_hessian(theta)
    return algopy.UTPM.extract_hessian(len(theta), f(theta))

def main():

    # read the data from the internet into numpy arrays
    medpar = pandas.read_csv(g_url)
    y_patsy, X_patsy = patsy.dmatrices('los~type2+type3+hmo+white', medpar)
    y = numpy.array(y_patsy).flatten()
    X = numpy.array(X_patsy)

    # define the objective function and the autodiff gradient and hessian
    f = functools.partial(get_neg_ll, y, X)
    g = functools.partial(eval_grad, f)
    h = functools.partial(eval_hess, f)

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
            avextol=1e-6,
            )

    # compute the hessian a couple of different ways
    algopy_hessian = h(results)
    num_hessian = numdifftools.Hessian(f)(results)

    # report the results of the search including aic and standard error
    print('search results:')
    print(results)
    print()
    print('aic:')
    print(get_aic(y, X, results))
    print()
    print('standard error using observed fisher information,')
    print('with hessian computed using algopy:')
    print(numpy.sqrt(numpy.diag(scipy.linalg.inv(algopy_hessian))))
    print()
    print('standard error using observed fisher information,')
    print('with hessian computed using numdifftools:')
    print(numpy.sqrt(numpy.diag(scipy.linalg.inv(num_hessian))))
    print()


if __name__ == '__main__':
    main()

