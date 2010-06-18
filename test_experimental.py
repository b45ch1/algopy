from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utpm import *
from algopy.globalfuncs import *

D,P,M,N = 2,9,30,10

# forward
A = UTPM(numpy.random.rand(D,P,M,N))
Q,R = UTPM.qr_full(A)
A2 = UTPM.dot(Q,R)
Q2, R2 = UTPM.qr_full(A2)

# reverse
Q2bar = UTPM(numpy.random.rand(D,P,M,M))
R2bar = UTPM.triu(UTPM(numpy.random.rand(D,P,M,N)))

A2bar = UTPM.pb_qr_full(Q2bar, R2bar, A2, Q2, R2)
Qbar, Rbar = UTPM.pb_dot(A2bar, Q, R, A2)
Abar = UTPM.pb_qr_full(Qbar, Rbar, A, Q, R)

# check forward calculation
assert_array_almost_equal(Q.data, Q2.data)
assert_array_almost_equal(R.data, R2.data)

# check reverse calculation: PART I
assert_array_almost_equal(Abar.data, A2bar.data)
assert_array_almost_equal( UTPM.triu(Rbar).data,  UTPM.triu(R2bar).data)
# cannot check Qbar and Q2bar since Q has only N*M - N(N+1)/2 distinct elements


# check reverse calculation: PART II
for p in range(P):
    Ab = Abar.data[0,p]
    Ad = A.data[1,p]

    Q2b = Q2bar.data[0,p]
    Q2d = Q2.data[1,p]

    R2b = R2bar.data[0,p]
    R2d = R2.data[1,p]

    assert_almost_equal(numpy.trace(numpy.dot(Ab.T,Ad)), numpy.trace(numpy.dot(Q2b.T,Q2d)) + numpy.trace(numpy.dot(R2b.T,R2d)))




