from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utp.utpm import *
from algopy.globalfuncs import *

D,P,M,N = 3,1,50,20

x = UTPM(numpy.random.rand(D,P,M,1))
A = dot(x,x.T)
# A = UTPM(numpy.random.rand(D,P,M,M))
# A[:,N:] = 0
Q,R = qr(A)


# print A - dot(Q,R)
# print dot(Q.T,Q) - numpy.eye(M)

Q2 = Q[:,N:]
# print dot(Q2.T, Q2)
print dot(A.T, Q2)

# print Q[:,N:]



    
    
    
    
