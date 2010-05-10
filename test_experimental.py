from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utp.utpm import *
from algopy.globalfuncs import *

D,P,M,N = 3,1,4,2

x = UTPM(numpy.random.rand(D,P,N,M))
A = dot(x.T,x)
# A = UTPM(numpy.random.rand(D,P,M,M))
# A.data[0,0,N:,:] = 10**-30
Q,R = qr(A)

# print A - dot(Q,R)
print dot(Q.T,Q) - numpy.eye(M)



    
    
    
    
