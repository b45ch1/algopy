from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utpm import *
from algopy.globalfuncs import *

D,P,M,N = 1,1,4,2
A = UTPM(numpy.random.rand(D,P,M,N))
Q,R = UTPM.qr_full(A)

assert_array_almost_equal(UTPM.triu(R).data,  R.data)
assert_array_almost_equal(A.data, UTPM.dot(Q,R).data)
assert_array_almost_equal(0, (UTPM.dot(Q.T,Q) - numpy.eye(M)).data)



Qbar = Q.zeros_like()
Rbar = R.zeros_like()


Abar = UTPM.pb_qr_full(Qbar, Rbar, A, Q, R)



