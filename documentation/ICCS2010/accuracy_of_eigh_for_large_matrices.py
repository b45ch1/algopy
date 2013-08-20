from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utpm import *

D,P,N = 4,1,100
A = UTPM(numpy.random.rand(D,P,N,N))
Q,R = UTPM.qr(A)
l = UTPM(numpy.random.rand(D,P,N))
L = UTPM.diag(l)

A = UTPM.dot(Q, UTPM.dot(L,Q.T))

l2,Q2 = UTPM.eigh(A)

print('error between true and reconstructed eigenvalues')
print(l.data[:,:,numpy.argsort(l.data[0,0])] - l2.data)



