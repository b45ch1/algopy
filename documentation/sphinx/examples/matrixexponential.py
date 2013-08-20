"""
This is an ML estimate of HKY85 molecular evolutionary parameters.
"""

import time

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
    """ function as sent in the email by alex """
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
    P = algopy.expm(Q)
    S = algopy.log(algopy.dot(algopy.diag(v), P))
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
    va = algopy.diag(algopy.sqrt(v))
    vb = algopy.diag(1./algopy.sqrt(v))
    W, U = algopy.eigh(algopy.dot(algopy.dot(va, Q), vb))
    M = algopy.dot(U, algopy.dot(algopy.diag(algopy.exp(W)), U.T))
    P = algopy.dot(vb, algopy.dot(M, va))
    S = algopy.log(algopy.dot(algopy.diag(v), P))
    return -algopy.sum(S * g_data)

def eval_grad_f_eigh(Y):
    """
    compute the gradient of f in the forward mode of AD
    """
    Y = algopy.UTPM.init_jacobian(Y)
    retval = eval_f_eigh(Y)
    return algopy.UTPM.extract_jacobian(retval)

def eval_hess_f_eigh(Y):
    """
    compute the hessian of f in the forward mode of AD
    """
    Y = algopy.UTPM.init_hessian(Y)
    retval = eval_f_eigh(Y)
    hessian = algopy.UTPM.extract_hessian(5, retval)
    return hessian

def eval_grad_f(Y):
    """
    compute the gradient of f in the forward mode of AD
    """
    Y = algopy.UTPM.init_jacobian(Y)
    retval = eval_f(Y)
    return algopy.UTPM.extract_jacobian(retval)

def eval_hess_f(Y):
    """
    compute the hessian of f in the forward mode of AD
    """
    Y = algopy.UTPM.init_hessian(Y)
    retval = eval_f(Y)
    hessian = algopy.UTPM.extract_hessian(5, retval)
    return hessian


def main():
    Y = np.zeros(5)


    print('--------------------------------')
    print(' simple check (functions)       ')
    print('--------------------------------')
    print('eval_f_orig(Y)\n', eval_f_orig(Y))
    print('eval_f(Y)\n', eval_f(Y))
    print('eval_f_eigh(Y)\n', eval_f_eigh(Y))
    print('eval_f_eigh(Y) - eval_f(Y)\n', eval_f_eigh(Y) - eval_f(Y))
    print('--------------------------------')
    print()


    print('--------------------------------')
    print(' simple check (gradients)       ')
    print('--------------------------------')
    print('eval_grad_f(Y)\n', eval_grad_f(Y))
    print('eval_grad_f_eigh(Y)\n', eval_grad_f_eigh(Y))
    print('eval_grad_f_eigh(Y) - eval_grad_f(Y)\n', eval_grad_f_eigh(Y) - eval_grad_f(Y))
    print('--------------------------------')
    print()


    print('--------------------------------')
    print(' simple check (hessians)        ')
    print('--------------------------------')
    print('eval_hess_f(Y)\n', eval_hess_f(Y))
    print('eval_hess_f_eigh(Y)\n', eval_hess_f_eigh(Y))
    print('eval_hess_f_eigh(Y) - eval_hess_f(Y)\n', eval_hess_f_eigh(Y) - eval_hess_f(Y))
    print('--------------------------------')
    print()

    tm = time.time()
    results = optimize.fmin(
            eval_f, Y,
            maxiter=10000, maxfun=10000, full_output=True)
    tsrate, tvrate, v = transform_params(results[0])
    print('--------------------------------')
    print('time:', time.time() - tm)
    print('results output from fmin:', results)
    print('estimated transition rate parameter:', tsrate)
    print('estimated transversion rate parameter:', tvrate)
    print('estimated stationary distribution:', v)
    print('--------------------------------')
    print()

    tm = time.time()
    results = optimize.fmin_ncg(
        #eval_f_eigh,             # obj. function
        eval_f,
        Y,                  # initial value
        #eval_grad_f_eigh,   # gradient of obj. function
        eval_grad_f,
        fhess_p=None,
        #fhess=eval_hess_f_eigh, # hessian of obj. function
        fhess=eval_hess_f,
        args=(),
        #avextol=1e-04,
        avextol=1e-05,
        epsilon=1.4901161193847656e-08,
        #maxiter=10,
        maxiter=100,
        full_output=True,
        disp=1,
        retall=0,
        callback=None)

    tsrate, tvrate, v = transform_params(results[0])
    #hess = eval_hess_f_eigh(results[0])
    hess = eval_hess_f(results[0])
    print('--------------------------------')
    print('time:', time.time() - tm)
    print('results output from fmin:', results)
    #print 'objective function value:', eval_f_eigh(results[0])
    print('objective function value:', eval_f(results[0]))
    #print 'gradient:', eval_grad_f_eigh(results[0])
    print('gradient:', eval_grad_f(results[0]))
    print('hessian:', hess)
    print('hess - hess.T:', hess - hess.T)
    print('eigvalsh(hess):', linalg.eigvalsh(hess))
    print('inverse of hessian:', linalg.inv(hess))
    print('determinant of hessian:', linalg.det(hess))
    print('estimated transition rate parameter:', tsrate)
    print('estimated transversion rate parameter:', tvrate)
    print('estimated stationary distribution:', v)
    print('--------------------------------')


if __name__ == '__main__':
    main()
