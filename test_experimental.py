from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utp.utpm import *
from algopy.globalfuncs import *

D,P,M,N = 3,1,5,2

A = UTPM(numpy.random.rand(D,P,M,M) + numpy.random.rand(D,P,M,M) * 1j)
B = UTPM(numpy.random.rand(D,P,M,M) + numpy.random.rand(D,P,M,M) * 1j)




print UTPM.dot(A,B)
    
    
    
