from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utpm import *
from algopy.globalfuncs import *

D,P,M,N = 3,1,5,2


A = UTPM(numpy.zeros((D,P,N,N)))

# A.data[0,0,0,0] = 1.
# A.data[1,0] = numpy.array([[1.,2.],[-1,-2]])

A.data[2,0] = numpy.random.rand(N,N)

# numpy.linalg.inv(numpy.zeros((2,2)))


print 'A=',A

Q,R = UTPM.qr(A)

# # print A
# # print R

print 'UTPM.dot(Q.T,Q)=',UTPM.dot(Q.T,Q)


print 'UTPM.dot(Q,R) - A=',UTPM.dot(Q,R) - A

    
    
    
