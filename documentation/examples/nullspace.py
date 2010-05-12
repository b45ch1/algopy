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
from algopy import CGraph, Function, UTPM, dot, qr, eigh, inv

# first order derivatives, one directional derivative
# D - 1 is the degree of the Taylor polynomial
# P directional derivatives at once
# M number of rows of J1
# N number of cols of J1
# K number of rows of J2 (must be smaller than N)
D,P,M,N,K,Nx = 2,1,100,3,1,1

J1 = UTPM(numpy.random.rand(*(D,P,M,N)))
J2 = UTPM(numpy.random.rand(*(D,P,K,N)))

# nullspace method
J2_tilde = UTPM(numpy.zeros((D,P,N,N)))
J2_tilde[:,:K] = J2.T
Q,R = qr(J2_tilde)
Q2 = Q[:,K:].T
J1_tilde = dot(J1,Q2.T)
Q,R = qr(J1_tilde)
tmp = inv(dot(R.T,R))
C = dot(Q2.T, dot(tmp,Q2))

print 'covariance matrix: C =\n',C
print 'check that Q2.T spans the nullspace of J2:\n', dot(J2,Q2.T)

# image space method
M = UTPM(numpy.zeros((D,P,N+K,N+K)))
M[:N,:N] = dot(J1.T,J1)
M[:N,N:] = J2.T
M[N:,:N] = J2
C2 = inv(M)[:N,:N]
print 'covariance matrix: C =\n',C2

print 'difference between image and nullspace method:\n',C - C2










