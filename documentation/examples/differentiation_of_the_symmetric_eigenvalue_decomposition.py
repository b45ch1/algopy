"""
This example shows how derivatives of all eigenvalue of a 100,100 matrix can be 
computed and compares it to the exact value.
"""

from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utpm import *
from algopy.globalfuncs import *

D,P,N = 4,1,100

print('create random matrix and perform QR decomposition ')
A = UTPM(numpy.random.rand(D,P,N,N))
Q,R = qr(A)

print('define diagonal matrix')
l = UTPM(numpy.random.rand(D,P,N))
L = diag(l)

print('transform to obtain non-diagonal matrix')
A = dot(Q, dot(L,Q.T))

print('reconstruct diagonal entries with eigh')
l2,Q2 = eigh(A)

print('absolute error true - reconstructed diagonal entries')
print(l.data[:,:,numpy.argsort(l.data[0,0])] - l2.data)

 
