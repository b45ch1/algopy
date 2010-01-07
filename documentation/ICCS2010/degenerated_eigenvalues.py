import numpy
from algopy.utp.utpm import *

D,P,N = 3,1,4
A = UTPM(numpy.zeros((D,P,N,N)))
V = UTPM(numpy.random.rand(D,P,N,N))

A.data[0,0] = numpy.diag([0,0,1,1.])


V,Rtilde = UTPM.qr(V)
A = UTPM.dot(UTPM.dot(V.T, A), V)


A0 = A.data[0,0]
l0,Q0 = numpy.linalg.eig(A0)

print l0

# print numpy.dot(Q0.T, Q0)
# print numpy.dot(Q0, Q0.T)


# print 


# L0 = numpy.diag(l0)
# B0 = numpy.dot(numpy.dot(Q0,L0), Q0.T)

# print A.data[0,0] - B0


# l,Q = UTPM.eig(A)
# L = UTPM.diag(l)


# B = UTPM.dot(Q, UTPM.dot(L,Q.T))

# print A - B
