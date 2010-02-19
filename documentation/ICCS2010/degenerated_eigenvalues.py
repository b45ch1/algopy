"""
This example shows that the push forward of the eigenvalue decomposition indeed
fails for symmetric matrices with degenerated eigenvalues.
"""

import numpy
from algopy.utp.utpm import *


# Build Symmetric Matrix with degenerate eigenvalues
D,P,N = 3,1,4
A = UTPM(numpy.zeros((D,P,N,N)))
V = UTPM(numpy.random.rand(D,P,N,N))
A.data[0,0] = numpy.diag([0,0,1,1.])
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
# L = UTPM.diag(l)

# print UTPM.dot(Q.T,Q)

# B = UTPM.dot(Q, UTPM.dot(L,Q.T))

# print A - B
