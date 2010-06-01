from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utpm import *
from algopy.globalfuncs import *


D,P,M,N = 2,1,6,3
A = UTPM(numpy.zeros((D,P,M,M)))
A[:,:N] = numpy.random.rand(M,N)

for m in range(M):
    for n in range(N):
        A.data[1] = 0.
        A.data[1,0,m,n] = 1.
        
        # STEP1: forward
        Q,R = qr(A)
        B = dot(Q,R)
        y = trace(B)
        
        # STEP2: reverse
        ybar = y.zeros_like()
        ybar.data[0,0] = 13./7.
        Bbar = UTPM.pb_trace(ybar, B, y)
        Qbar, Rbar = UTPM.pb_dot(Bbar, Q, R, B)
        Abar = UTPM.pb_qr(Qbar, Rbar, A, Q, R)
        
        assert_array_almost_equal((m==n)*13./7., Abar.data[0,0,m,n])


# Rbar.data[:,:,:,:] = numpy.random.rand(D,P,M,M)

# assert_array_almost_equal(UTPM.triu(R).data,  R.data)

# print Abar

# print A - B



# for d in range(D):
#     for p in range(P):
#         print (R.data[d,p] - numpy.triu(R.data[d,p])).argmax()
#         # assert_array_almost_equal(R.data[d,p], numpy.triu(R.data[d,p]))

# assert_array_almost_equal(A.data, UTPM.dot(Q,R).data)
# assert_array_almost_equal(0, (UTPM.dot(Q.T,Q) - numpy.eye(M)).data)

# Q2 = Q[:,N:]
# # assert_array_almost_equal(0, UTPM.dot(A.T, Q2).data)

# print UTPM.dot(A.T, Q2).data.max()

    
    
    
