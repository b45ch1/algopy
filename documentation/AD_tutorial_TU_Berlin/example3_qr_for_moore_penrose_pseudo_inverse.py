"""
This is a simple test that differentiates the Moore-Penrose pseudo-inverse 
computation. Explicitly:

..math::
    A^\dagger = (A^T A)^{-1} A^T

Two methods are compared. First the naive approach by first computing A^T A,
then invert it and then multiplication with A^T.
Then the QR approach.

"""


import numpy
from algopy import CGraph, Function, UTPM, dot, qr, eigh, inv, solve

D,P,M,N = 2,1,5,2

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

# Method 1: Naive approach
Apinv = dot(inv(dot(A.T,A)),A.T)

print('naive approach: A Apinv A - A = 0 \n', dot(dot(A, Apinv),A) - A)
print('naive approach: Apinv A Apinv - Apinv = 0 \n', dot(dot(Apinv, A),Apinv) - Apinv)
print('naive approach: (Apinv A)^T - Apinv A = 0 \n', dot(Apinv, A).T  - dot(Apinv, A))
print('naive approach: (A Apinv)^T - A Apinv = 0 \n', dot(A, Apinv).T  - dot(A, Apinv))


# Method 2: Using the differentiated QR decomposition
Q,R = qr(A)
tmp1 = solve(R.T, A.T)
tmp2 = solve(R, tmp1)
Apinv = tmp2

print('QR approach: A Apinv A - A = 0 \n',  dot(dot(A, Apinv),A) - A)
print('QR approach: Apinv A Apinv - Apinv = 0 \n', dot(dot(Apinv, A),Apinv) - Apinv)
print('QR approach: (Apinv A)^T - Apinv A = 0 \n', dot(Apinv, A).T  - dot(Apinv, A))
print('QR approach: (A Apinv)^T - A Apinv = 0 \n', dot(A, Apinv).T  - dot(A, Apinv)) 
