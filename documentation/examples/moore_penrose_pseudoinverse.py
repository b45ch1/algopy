"""
In this example it is the goal to compute derivatives of the Moore-Penrose pseudoinverse
in a stable manner.

I.e. compute::

    A^\dagger = (A^T A)^{-1} A^T

where A is a (M,N) array (M >= N) with possibly bad condition number
"""

import numpy
from algopy import CGraph, Function, UTPM, dot, qr, eigh, inv, solve

# first order derivatives, one directional derivative
# D - 1 is the degree of the Taylor polynomial
# P directional derivatives at once
# M number of rows of J1
# N number of cols of J1
# K number of rows of J2 (must be smaller than N)
D,P,M,N,K,Nx = 2,1,5,2,1,1


# generate badly conditioned matrix A
A = UTPM(numpy.zeros((D,P,M,N)))
x = UTPM(numpy.zeros((D,P,M,1)))
y = UTPM(numpy.zeros((D,P,M,1)))

x.data[0,0,:,0] = [1,1,1,1,1]
x.data[1,0,:,0] = [1,1,1,1,1]

y.data[0,0,:,0] = [1,2,1,2,1]
y.data[1,0,:,0] = [1,2,1,2,1]


alpha = 10**-5
A = dot(x,x.T) + alpha*dot(y,y.T)

A = A[:,:2]
Apinv = dot(inv(dot(A.T,A)),A.T)

print 'naive approach: A Apinv A - A = 0 \n', dot(dot(A, Apinv),A) - A
print 'naive approach: Apinv A Apinv - Apinv = 0 \n', dot(dot(Apinv, A),Apinv) - Apinv
print 'naive approach: (Apinv A)^T - Apinv A = 0 \n', dot(Apinv, A).T  - dot(Apinv, A)
print 'naive approach: (A Apinv)^T - A Apinv = 0 \n', dot(A, Apinv).T  - dot(A, Apinv)

# print dot(C,B)

Q,R = qr(A)
tmp1 = solve(R.T, A.T)
tmp2 = solve(R, tmp1)
Apinv = tmp2

print 'QR approach: A Apinv A - A = 0 \n',  dot(dot(A, Apinv),A) - A
print 'QR approach: Apinv A Apinv - Apinv = 0 \n', dot(dot(Apinv, A),Apinv) - Apinv
print 'QR approach: (Apinv A)^T - Apinv A = 0 \n', dot(Apinv, A).T  - dot(Apinv, A)
print 'QR approach: (A Apinv)^T - A Apinv = 0 \n', dot(A, Apinv).T  - dot(A, Apinv)


# print numpy.linalg.svd(Apinv.data[0,0])[1]

# print 'dot(A, Apinv)=',dot(A, Apinv)

