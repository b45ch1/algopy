# coding: utf8
# from __future__ import division
import numpy as np
from numpy.linalg import norm
import algopy


def gauss_algopy(eval_f, x0, tol=10e-5):

    x0 = np.array(x0, dtype=float)
    cg = algopy.CGraph()
    x = algopy.Function(x0)
    y = eval_f(x)
    cg.trace_off()
    cg.independentFunctionList = [x]
    cg.dependentFunctionList = [y]
    sol = gauss_solver(eval_f, cg.jacobian, x0, tol)
    return sol

def gauss_solver(f, jac, x0, tol):

    i=0
    x = x0
    res=1
    res_alt=10e10
    while((res>tol) and (50>i) and (abs(res-res_alt)>tol**2)):

        i+=1
        r=np.matrix(f(x))

        D = np.matrix(jac(x))
        DD=np.linalg.solve(D.T*D,D.T*r.T)

        x = np.matrix(x).T - DD
        x= np.array(x).flatten().tolist()
        res_alt=res
        res = norm(r)
        print(i,': ',res)

    return x


if __name__ == '__main__':

    def eval_f1(x):
        y = algopy.zeros(3, dtype=x)
        y[0] = x[1] + x[2] - 5.
        y[1] = -x[0] + x[2]**3 + 1.
        y[2] = -x[1] + algopy.tan(x[0]) -x[2] + 7.
        return y

    def eval_f2(x):
        y = algopy.zeros(3, dtype=x)
        y[0] = x[1] + x[2] - 5.
        y[1] = -1./x[0] + x[2]**3 + 1.
        y[2] = -x[1] + algopy.tan(x[0]) -x[2] + 7.
        return y

    x0 = np.array([100.0, 1.0, 1.0])
    sol = gauss_algopy(eval_f1, x0)
    print(sol)

    x0 = np.array([100.0, 1.0, 1.0])
    sol = gauss_algopy(eval_f2, x0)
    print(sol)
