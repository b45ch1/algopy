from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utpm import *
from algopy.globalfuncs import *

D,P,M,N = 3,1,4,2
A = UTPM(numpy.random.rand(D,P,M,M))
A.data[:,0,N:,:] = 0
Q,R = UTPM.qr(A)

print R


assert_array_almost_equal(UTPM.triu(R).data,  R.data)
assert_array_almost_equal(A.data, UTPM.dot(Q,R).data)
assert_array_almost_equal(0, (UTPM.dot(Q.T,Q) - numpy.eye(M)).data)
