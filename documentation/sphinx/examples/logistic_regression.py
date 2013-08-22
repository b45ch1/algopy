"""
Logistic Regression

This is an algopy implementation of a scicomp stackexchange question
http://scicomp.stackexchange.com/questions/4826/logistic-regression-with-python
"""

import functools

import numpy
import scipy.optimize
import algopy


g_data = """\
low,age,lwt,race,smoke,ptl,ht,ui
0,19,182,2,0,0,0,1
0,33,155,3,0,0,0,0
0,20,105,1,1,0,0,0
0,21,108,1,1,0,0,1
0,18,107,1,1,0,0,1
0,21,124,3,0,0,0,0
0,22,118,1,0,0,0,0
0,17,103,3,0,0,0,0
0,29,123,1,1,0,0,0
0,26,113,1,1,0,0,0
0,19,95,3,0,0,0,0
0,19,150,3,0,0,0,0
0,22,95,3,0,0,1,0
0,30,107,3,0,1,0,1
0,18,100,1,1,0,0,0
0,18,100,1,1,0,0,0
0,15,98,2,0,0,0,0
0,25,118,1,1,0,0,0
0,20,120,3,0,0,0,1
0,28,120,1,1,0,0,0
0,32,121,3,0,0,0,0
0,31,100,1,0,0,0,1
0,36,202,1,0,0,0,0
0,28,120,3,0,0,0,0
0,25,120,3,0,0,0,1
0,28,167,1,0,0,0,0
0,17,122,1,1,0,0,0
0,29,150,1,0,0,0,0
0,26,168,2,1,0,0,0
0,17,113,2,0,0,0,0
0,17,113,2,0,0,0,0
0,24,90,1,1,1,0,0
0,35,121,2,1,1,0,0
0,25,155,1,0,0,0,0
0,25,125,2,0,0,0,0
0,29,140,1,1,0,0,0
0,19,138,1,1,0,0,0
0,27,124,1,1,0,0,0
0,31,215,1,1,0,0,0
0,33,109,1,1,0,0,0
0,21,185,2,1,0,0,0
0,19,189,1,0,0,0,0
0,23,130,2,0,0,0,0
0,21,160,1,0,0,0,0
0,18,90,1,1,0,0,1
0,18,90,1,1,0,0,1
0,32,132,1,0,0,0,0
0,19,132,3,0,0,0,0
0,24,115,1,0,0,0,0
0,22,85,3,1,0,0,0
0,22,120,1,0,0,1,0
0,23,128,3,0,0,0,0
0,22,130,1,1,0,0,0
0,30,95,1,1,0,0,0
0,19,115,3,0,0,0,0
0,16,110,3,0,0,0,0
0,21,110,3,1,0,0,1
0,30,153,3,0,0,0,0
0,20,103,3,0,0,0,0
0,17,119,3,0,0,0,0
0,17,119,3,0,0,0,0
0,23,119,3,0,0,0,0
0,24,110,3,0,0,0,0
0,28,140,1,0,0,0,0
0,26,133,3,1,2,0,0
0,20,169,3,0,1,0,1
0,24,115,3,0,0,0,0
0,28,250,3,1,0,0,0
0,20,141,1,0,2,0,1
0,22,158,2,0,1,0,0
0,22,112,1,1,2,0,0
0,31,150,3,1,0,0,0
0,23,115,3,1,0,0,0
0,16,112,2,0,0,0,0
0,16,135,1,1,0,0,0
0,18,229,2,0,0,0,0
0,25,140,1,0,0,0,0
0,32,134,1,1,1,0,0
0,20,121,2,1,0,0,0
0,23,190,1,0,0,0,0
0,22,131,1,0,0,0,0
0,32,170,1,0,0,0,0
0,30,110,3,0,0,0,0
0,20,127,3,0,0,0,0
0,23,123,3,0,0,0,0
0,17,120,3,1,0,0,0
0,19,105,3,0,0,0,0
0,23,130,1,0,0,0,0
0,36,175,1,0,0,0,0
0,22,125,1,0,0,0,0
0,24,133,1,0,0,0,0
0,21,134,3,0,0,0,0
0,19,235,1,1,0,1,0
0,25,95,1,1,3,0,1
0,16,135,1,1,0,0,0
0,29,135,1,0,0,0,0
0,29,154,1,0,0,0,0
0,19,147,1,1,0,0,0
0,19,147,1,1,0,0,0
0,30,137,1,0,0,0,0
0,24,110,1,0,0,0,0
0,19,184,1,1,0,1,0
0,24,110,3,0,1,0,0
0,23,110,1,0,0,0,0
0,20,120,3,0,0,0,0
0,25,241,2,0,0,1,0
0,30,112,1,0,0,0,0
0,22,169,1,0,0,0,0
0,18,120,1,1,0,0,0
0,16,170,2,0,0,0,0
0,32,186,1,0,0,0,0
0,18,120,3,0,0,0,0
0,29,130,1,1,0,0,0
0,33,117,1,0,0,0,1
0,20,170,1,1,0,0,0
0,28,134,3,0,0,0,0
0,14,135,1,0,0,0,0
0,28,130,3,0,0,0,0
0,25,120,1,0,0,0,0
0,16,95,3,0,0,0,0
0,20,158,1,0,0,0,0
0,26,160,3,0,0,0,0
0,21,115,1,0,0,0,0
0,22,129,1,0,0,0,0
0,25,130,1,0,0,0,0
0,31,120,1,0,0,0,0
0,35,170,1,0,1,0,0
0,19,120,1,1,0,0,0
0,24,116,1,0,0,0,0
0,45,123,1,0,0,0,0
1,28,120,3,1,1,0,1
1,29,130,1,0,0,0,1
1,34,187,2,1,0,1,0
1,25,105,3,0,1,1,0
1,25,85,3,0,0,0,1
1,27,150,3,0,0,0,0
1,23,97,3,0,0,0,1
1,24,128,2,0,1,0,0
1,24,132,3,0,0,1,0
1,21,165,1,1,0,1,0
1,32,105,1,1,0,0,0
1,19,91,1,1,2,0,1
1,25,115,3,0,0,0,0
1,16,130,3,0,0,0,0
1,25,92,1,1,0,0,0
1,20,150,1,1,0,0,0
1,21,200,2,0,0,0,1
1,24,155,1,1,1,0,0
1,21,103,3,0,0,0,0
1,20,125,3,0,0,0,1
1,25,89,3,0,2,0,0
1,19,102,1,0,0,0,0
1,19,112,1,1,0,0,1
1,26,117,1,1,1,0,0
1,24,138,1,0,0,0,0
1,17,130,3,1,1,0,1
1,20,120,2,1,0,0,0
1,22,130,1,1,1,0,1
1,27,130,2,0,0,0,1
1,20,80,3,1,0,0,1
1,17,110,1,1,0,0,0
1,25,105,3,0,1,0,0
1,20,109,3,0,0,0,0
1,18,148,3,0,0,0,0
1,18,110,2,1,1,0,0
1,20,121,1,1,1,0,1
1,21,100,3,0,1,0,0
1,26,96,3,0,0,0,0
1,31,102,1,1,1,0,0
1,15,110,1,0,0,0,0
1,23,187,2,1,0,0,0
1,20,122,2,1,0,0,0
1,24,105,2,1,0,0,0
1,15,115,3,0,0,0,1
1,23,120,3,0,0,0,0
1,30,142,1,1,1,0,0
1,22,130,1,1,0,0,0
1,17,120,1,1,0,0,0
1,23,110,1,1,1,0,0
1,17,120,2,0,0,0,0
1,26,154,3,0,1,1,0
1,20,106,3,0,0,0,0
1,26,190,1,1,0,0,0
1,14,101,3,1,1,0,0
1,28,95,1,1,0,0,0
1,14,100,3,0,0,0,0
1,23,94,3,1,0,0,0
1,17,142,2,0,0,1,0
1,21,130,1,1,0,1,0
"""


########################################################################
# boilerplate functions for algopy

def eval_grad(f, theta):
    theta = algopy.UTPM.init_jacobian(theta)
    return algopy.UTPM.extract_jacobian(f(theta))

def eval_hess(f, theta):
    theta = algopy.UTPM.init_hessian(theta)
    return algopy.UTPM.extract_hessian(len(theta), f(theta))


########################################################################
# application specific functions

def get_neg_ll(vY, mX, vBeta):
    """
    @param vY: predefined numpy array
    @param mX: predefined numpy array
    @param vBeta: parameters of the likelihood function
    """
    #FIXME: algopy could benefit from the addition of a logsumexp function...
    alpha = algopy.dot(mX, vBeta)
    return algopy.sum(
            vY*algopy.log1p(algopy.exp(-alpha)) +
            (1-vY)*algopy.log1p(algopy.exp(alpha)))

def preprocess_data():
    """
    Convert the data from a hardcoded string into something nicer.
    """
    data = numpy.loadtxt(g_data.splitlines(), delimiter=',', skiprows=1)
    vY = data[:, 0]
    mX = data[:, 1:]
    intercept = numpy.ones(mX.shape[0]).reshape(mX.shape[0], 1)
    mX = numpy.concatenate((intercept, mX), axis=1)
    iK = mX.shape[1]
    iN = mX.shape[0]
    return vY, mX

def main():

    # extract the data from the hardcoded string
    vY, mX = preprocess_data()

    # this is a hardcoded point which is supposed to have a good likelihood
    vBeta_star = numpy.array([
        -.10296645, -.0332327, -.01209484, .44626211,
        .92554137, .53973828, 1.7993371, .7148045,
        ])

    # these are arbitrary parameter values somwhat near the good values
    vBeta_0 = numpy.array([-.1, -.03, -.01, .44, .92, .53, 1.8, .71])

    # define the objective function and the autodiff gradient and hessian
    f = functools.partial(get_neg_ll, vY, mX)
    g = functools.partial(eval_grad, f)
    h = functools.partial(eval_hess, f)

    # show the neg log likelihood for the good parameters
    print('hardcoded good values:')
    print(vBeta_star)
    print()
    print('neg log likelihood for good values:')
    print(f(vBeta_star))
    print()
    print()
    print('hardcoded okay values:')
    print(vBeta_0)
    print()
    print('neg log likelihood for okay values:')
    print(f(vBeta_0))
    print()
    print()

    # do the max likelihood search
    results = scipy.optimize.fmin_ncg(
            f,
            vBeta_0,
            fprime=g,
            fhess=h,
            avextol=1e-6,
            disp=0,
            )

    # extract the max likelihood values
    vBeta_mle = results

    print('maximum likelihood estimates:')
    print(vBeta_mle)
    print()
    print('neg log likelihood for maximum likelihood estimates:')
    print(f(vBeta_mle))
    print()


if __name__ == '__main__':
    main()

