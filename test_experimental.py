from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utp.utpm import *
from algopy.globalfuncs import *

D,P,N = 2,3,2
tmp = numpy.random.rand(D,P,N,N) - 0.5
tmp[0,:] = tmp[0,0]

X = UTPM(tmp)

assert_array_almost_equal(X.data, 0.5*( abs(X + abs(X)) - abs(X - abs(X))).data)




    
    
    
    
