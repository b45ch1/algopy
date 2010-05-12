from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utpm import *
from algopy.globalfuncs import *

D,P,M,N = 3,1,5,2


A = UTPM(numpy.zeros((D,P,N,N)))

A.data[1,0] = numpy.ones((N,N))

# numpy.linalg.inv(numpy.zeros((2,2)))

Q,R = UTPM.qr(A)

# print A
print R

# print UTPM.dot(Q.T,Q)


# print UTPM.dot(Q,R) - A

    
    
    
