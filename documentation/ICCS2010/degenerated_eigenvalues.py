"""
This example shows that the push forward of the eigenvalue decomposition indeed
fails for symmetric matrices with degenerated eigenvalues.
"""

import numpy
from algopy.utpm import *


# Build Symmetric Matrix with degenerate eigenvalues
D,P,N = 3,1,4
A = UTPM(numpy.zeros((D,P,N,N)))
V = UTPM(numpy.random.rand(D,P,N,N))

# A.data[0,0] = numpy.diag([2,2,3,3,2,5.])
# A.data[1,0] = numpy.diag([5,5,3,1,1,3.])
# A.data[2,0] = numpy.diag([3,1,3,1,1,3.])

A.data[0,0] = numpy.diag([2,2,2,5.])
A.data[1,0] = numpy.diag([5,5,6,6.])
A.data[2,0] = numpy.diag([1,1,1,1.])


V,Rtilde = UTPM.qr(V)
A = UTPM.dot(UTPM.dot(V.T, A), V)

# sanity check: solution of the zero'th coefficient using numpy
A0 = A.data[0,0]
l0,Q0 = numpy.linalg.eigh(A0)
L0 = numpy.diag(l0)
B0 = numpy.dot(numpy.dot(Q0,L0), Q0.T)

# pushforward: general UTPM solution
l,Q = UTPM.eigh(A)
L = UTPM.diag(l)

# pullback
lbar = UTPM(numpy.random.rand(*(D,P,N)))
Qbar = UTPM(numpy.random.rand(*(D,P,N,N)))
Abar = UTPM.pb_eigh( lbar, Qbar, A, l, Q)

Abar = Abar.data[0,0]
Adot = A.data[1,0]

Lbar = UTPM._diag(lbar.data)[0,0]
Ldot = UTPM._diag(l.data)[1,0]

Qbar = Qbar.data[0,0]
Qdot = Q.data[1,0]

# print l

# print 'check pushforward:'
print('Q.T A Q - L =\n', UTPM.dot(Q.T, UTPM.dot(A,Q)) - L)
# print 'Q.T Q - I =\n', UTPM.dot(Q.T, Q) - numpy.eye(N)
# print 'check pullback:'
# print 'error measure of the pullback = ', numpy.trace(numpy.dot(Abar.T, Adot)) - numpy.trace( numpy.dot(Lbar.T, Ldot) + numpy.dot(Qbar.T, Qdot))

