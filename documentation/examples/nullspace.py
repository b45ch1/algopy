"""
This example computes the covariance matrix of a constrained parameter estimation
problem by a nullspace method.

I.e. compute

            / J1.T J1    J2.T  \^-1  / 1 \
C =  (1,0)  |                  |     |   |
            \ J2          0    /     \ 0 /

where J1 = J1(x) and J2 = J2(x), where x is Nx dimensional.
Goal is the numerically computation (this claim is not thoroughly tested yet!),
i.e. it is advicable not to multiply J1.T J1 since this would square the condition
number.

"""

import numpy
from algopy import CGraph, Function, UTPM, dot, qr, eigh

# first order derivatives, one directional derivative
# D - 1 is the degree of the Taylor polynomial
# P directional derivatives at once
# M number of rows of J1
# N number of cols of J1
# K number of rows of J2 (must be smaller than N)
D,P,M,N,K,Nx = 2,1,10,3,2,1

J1 = UTPM(numpy.random.rand(*(D,P,M,N)))
J2 = UTPM(numpy.random.rand(*(D,P,K,N)))

J2_tilde = UTPM(numpy.zeros((D,P,N,N)))
J2_tilde[:,:K] = J2.T

Q,R = qr(J2_tilde)

Q2 = Q[:,K:].T

print 'check that Q2.T spans the nullspace of J2:', dot(J2,Q2.T)
J1_tilde = dot(J1,Q2.T)
Q,R = qr(J1_tilde)

tmp = dot(R.T,R)
C = dot(Q2.T, dot(tmp,Q2))
print 'covariance matrix: C =',C






