"""
In this example it is the goal to compute derivatives of the covariance matrix
of a constrained parameter estimation problem.

I.e. compute

            / J1.T J1    J2.T  \^-1  / 1 \
C =  (1,0)  |                  |     |   |
            \ J2          0    /     \ 0 /

where J1 = J1(x) and J2 = J2(x). The vector x is Nx dimensional.

Two possibilities are compared:
    1) filling a big matrix with elements, then invert it and return a view of
       of the upper left part of the matrix
    
    2) Computation of the Nullspace of J2 with a QR decomposition.
       The formula is::
           C = Q2.T( Q2 J1.T J1 Q2.T)^-1 Q2 .
       Potentially, using the QR decomposition twice, i.e. once to compute Q2 and
       then for J1 Q2.T to avoid the multiplication which would square the condition
       number, may be numerically more stable. This has not been tested yet though.
       This example only shows the computation of Q2.

I.e. this small example is a demonstration how the QR decomposition on polynomial
matrices can be used.
"""

import numpy
from algopy import CGraph, Function, UTPM, dot, qr, eigh, inv, solve

# first order derivatives, one directional derivative
# D - 1 is the degree of the Taylor polynomial
# P directional derivatives at once
# M number of rows of J1
# N number of cols of J1
# K number of rows of J2 (must be smaller than N)
D,P,M,N,K,Nx = 2,1,100,3,1,1


# METHOD 1: nullspace method
cg1 = CGraph()

J1 = Function(UTPM(numpy.random.rand(*(D,P,M,N))))
J2 = Function(UTPM(numpy.random.rand(*(D,P,K,N))))


Q,R = Function.qr_full(J2.T)
Q2 = Q[:,K:].T

J1_tilde = dot(J1,Q2.T)
Q,R = qr(J1_tilde)
V = solve(R.T, Q2)
C = dot(V.T,V)
cg1.trace_off()

cg1.independentFunctionList = [J1, J2]
cg1.dependentFunctionList = [C]

print('covariance matrix: C =\n',C)
print('check that Q2.T spans the nullspace of J2:\n', dot(J2,Q2.T))

# METHOD 2: image space method (potentially numerically unstable)
cg2 = CGraph()

J1 = Function(J1.x)
J2 = Function(J2.x)

M = Function(UTPM(numpy.zeros((D,P,N+K,N+K))))
M[:N,:N] = dot(J1.T,J1)
M[:N,N:] = J2.T
M[N:,:N] = J2
C2 = inv(M)[:N,:N]
cg2.trace_off()

cg2.independentFunctionList = [J1, J2]
cg2.dependentFunctionList = [C2]

print('covariance matrix: C =\n',C2)
print('difference between image and nullspace method:\n',C - C2)

Cbar = UTPM(numpy.random.rand(D,P,N,N))

cg1.pullback([Cbar])

cg2.pullback([Cbar])
print('J1\n',cg2.independentFunctionList[0].xbar - cg1.independentFunctionList[0].xbar)
print('J2\n',cg2.independentFunctionList[1].xbar - cg1.independentFunctionList[1].xbar)














