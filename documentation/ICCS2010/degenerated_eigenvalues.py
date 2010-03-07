"""
This example shows that the push forward of the eigenvalue decomposition indeed
fails for symmetric matrices with degenerated eigenvalues.
"""

import numpy
from algopy.utp.utpm import *


# Build Symmetric Matrix with degenerate eigenvalues
D,P,N = 2,1,4
A = UTPM(numpy.zeros((D,P,N,N)))
V = UTPM(numpy.random.rand(D,P,N,N))

A.data[0,0] = numpy.diag([2,2,3,3.])
A.data[1,0] = numpy.diag([5,6,7,7.])

V,Rtilde = UTPM.qr(V)
A = UTPM.dot(UTPM.dot(V.T, A), V)

# solution of the zero'th coefficient using numpy
A0 = A.data[0,0]
l0,Q0 = numpy.linalg.eigh(A0)
# print numpy.dot(Q0, Q0.T)
L0 = numpy.diag(l0)
B0 = numpy.dot(numpy.dot(Q0,L0), Q0.T)
# print A.data[0,0] - B0

# general UTPM solution
l,Q = UTPM.eigh(A)
L = UTPM.diag(l)

# Q1 = Q[:,:2]

# print UTPM.dot(Q1.T, UTPM.dot(A,Q1))

tmp2 = numpy.dot(Q.T.data[1,0], numpy.dot(A.data[0,0], Q.data[0,0]))
tmp3 = numpy.dot(Q.T.data[0,0], numpy.dot(A.data[1,0], Q.data[0,0]))
tmp4 = numpy.dot(Q.T.data[0,0], numpy.dot(A.data[0,0], Q.data[1,0]))

tmp = tmp2 + tmp3 + tmp4

# print tmp
U = numpy.linalg.eigh(tmp)[1]
# U[2:,:2] = 0.
# U[:2,2:] = 0.
# print numpy.dot(U.T,U)

Q = UTPM.dot(Q, U)

# print UTPM.dot(Q.T, Q)

print UTPM.dot(Q.T, UTPM.dot(A,Q))



# print tmp1 - tmp2


# B = UTPM.dot(Q, UTPM.dot(L,Q.T))

# print A - B
