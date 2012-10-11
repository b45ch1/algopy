"""
This is an ML estimate of HKY85 molecular evolutionary parameters.

Author: alex <argriffi@ncsu.edu>

"""

import numpy as np
import algopy
from scipy import optimize, linalg

# AGCT subsititution counts between human and chimp mitochondrial coding dna.
g_data = np.array([
        [2954, 141, 17, 16],
        [165, 1110, 5, 2],
        [18, 4, 3163, 374],
        [15, 2, 310, 2411],
        ],dtype=float)

def transform_params(Y):
    X = algopy.exp(Y)
    tsrate, tvrate = X[0], X[1]
    v_unnormalized = algopy.zeros(4, dtype=X)
    v_unnormalized[0] = X[2]
    v_unnormalized[1] = X[3]
    v_unnormalized[2] = X[4]
    v_unnormalized[3] = 1.0
    v = v_unnormalized / algopy.sum(v_unnormalized)
    return tsrate, tvrate, v

def eval_f_orig(Y):
    """ function as sent in the email by alex <argriffi@ncsu.edu> """
    a, b, v = transform_params(Y)
    Q = np.array([
        [0, a, b, b],
        [a, 0, b, b],
        [b, b, 0, a],
        [b, b, a, 0],
        ])
    Q = np.dot(Q, np.diag(v))
    Q -= np.diag(np.sum(Q, axis=1))
    S = np.log(np.dot(np.diag(v), linalg.expm(Q)))
    return -np.sum(S * g_data)

def eval_f(Y):
    """ some reformulations to make eval_f_orig
        compatible with algopy

        missing: support for scipy.linalg.expm

        i.e., this function can't be differentiated with algopy

    """

    a, b, v = transform_params(Y)

    Q = algopy.zeros((4,4), dtype=Y)
    Q[0,0] = 0;    Q[0,1] = a;    Q[0,2] = b;    Q[0,3] = b;
    Q[1,0] = a;    Q[1,1] = 0;    Q[1,2] = b;    Q[1,3] = b;
    Q[2,0] = b;    Q[2,1] = b;    Q[2,2] = 0;    Q[2,3] = a;
    Q[3,0] = b;    Q[3,1] = b;    Q[3,2] = a;    Q[3,3] = 0;

    Q = Q * v
    Q -= algopy.diag(algopy.sum(Q, axis=1))
    B = linalg.expm(Q)
    S = algopy.log(algopy.dot(algopy.diag(v), B))
    return -algopy.sum(S * g_data)

def eval_f_eigh(Y):
    """ some reformulations to make eval_f_orig
        compatible with algopy

        replaced scipy.linalg.expm by a symmetric eigenvalue decomposition

        this function **can** be differentiated with algopy

    """
    a, b, v = transform_params(Y)

    Q = algopy.zeros((4,4), dtype=Y)
    Q[0,0] = 0;    Q[0,1] = a;    Q[0,2] = b;    Q[0,3] = b;
    Q[1,0] = a;    Q[1,1] = 0;    Q[1,2] = b;    Q[1,3] = b;
    Q[2,0] = b;    Q[2,1] = b;    Q[2,2] = 0;    Q[2,3] = a;
    Q[3,0] = b;    Q[3,1] = b;    Q[3,2] = a;    Q[3,3] = 0;

    Q = algopy.dot(Q, algopy.diag(v))
    Q -= algopy.diag(algopy.sum(Q, axis=1))
    d,U = algopy.eigh(Q)
    d = algopy.exp(d)
    B = algopy.dot(U*d, U.T)
    S = algopy.log(algopy.dot(algopy.diag(v), B))
    return -algopy.sum(S * g_data)

def eval_grad_f_eigh(Y):
    """
    compute the gradient of f in the forward mode of AD
    """

    Y = algopy.UTPM.init_jacobian(Y)
    retval = eval_f_eigh(Y)
    return algopy.UTPM.extract_jacobian(retval)

def main():
    Y = np.zeros(5)

    print '--------------------------------'
    print eval_f_orig(Y)
    print eval_f(Y)
    print eval_f_eigh(Y)
    print eval_grad_f_eigh(Y)
    print '--------------------------------'
    print

    results = optimize.fmin(
            eval_f, Y,
            maxiter=10000, maxfun=10000, full_output=True)
    tsrate, tvrate, v = transform_params(results[0])
    print '--------------------------------'
    print 'results output from fmin:', results
    print 'estimated transition rate parameter:', tsrate
    print 'estimated transversion rate parameter:', tvrate
    print 'estimated stationary distribution:', v
    print '--------------------------------'
    print

    results = optimize.fmin_ncg(
        eval_f,             # obj. function
        Y,                  # initial value
        eval_grad_f_eigh,   # gradient of obj. function
        fhess_p=None,
        fhess=None,
        args=(),
        avextol=1e-07,
        epsilon=1.4901161193847656e-08,
        maxiter=10000,
        full_output=True,
        disp=1,
        retall=0,
        callback=None)

    tsrate, tvrate, v = transform_params(results[0])
    print '--------------------------------'
    print 'results output from fmin:', results
    print 'estimated transition rate parameter:', tsrate
    print 'estimated transversion rate parameter:', tvrate
    print 'estimated stationary distribution:', v
    print '--------------------------------'


if __name__ == '__main__':
    main()
