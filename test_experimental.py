from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utpm import *
from algopy.globalfuncs import *


A = UTPM(numpy.zeros((5,1,3,3)))
A.data[0,0] = [[1.000000000000001e+00, 5.000000000000002e-01  ,5.551115123125785e-17],
[9.999999999999999e-01  ,   1.000000000000000e+00  ,  -6.162975822039155e-33],
[1.110223024625157e-16  ,   1.000000000000000e+00  ,   6.162975822039155e-33]]

A.data[1,0] = [[0   , -2.500000000000003e-01  ,  -2.775557561562893e-17],
[0 ,    2.775557561562894e-17  ,   3.081487911019577e-33],
[0  ,  -2.775557561562894e-17  ,  -3.081487911019577e-33]]

A.data[2,0] =[[0,     1.250000000000001e-01,     1.387778780781446e-17],
[0 ,   -1.387778780781446e-17 ,   -1.540743955509789e-33],
[0  ,   1.387778780781446e-17 ,    1.540743955509789e-33]]


A.data[3,0] = [[0 ,   -6.250000000000004e-02  ,  -6.938893903907231e-18],
[0   ,  6.938893903907230e-18  ,   7.703719777548943e-34],
[0   , -6.938893903907230e-18  ,  -7.703719777548943e-34]]

A.data[4,0] = [[0  ,   3.125000000000002e-02  ,   3.469446951953616e-18],
[0  ,  -3.469446951953615e-18  ,  -3.851859888774472e-34],
[0   ,  3.469446951953615e-18  ,   3.851859888774472e-34]]


Q,R = UTPM.qr(A)

print 'UTPM.dot(Q.T,Q)=\n',UTPM.dot(Q.T,Q)
print 'R =\n',R
print 'UTPM.dot(Q,R) - A=\n',UTPM.dot(Q,R) - A



# assert_array_almost_equal(UTPM.triu(R).data,  R.data)
# assert_array_almost_equal(A.data, UTPM.dot(Q,R).data)
# assert_array_almost_equal(0, (UTPM.dot(Q.T,Q) - numpy.eye(M)).data)


# D,P,M,N = 3,1,6,3
# A = UTPM(numpy.random.rand(D,P,M,M))
# A[:,N:] = 0
# Q,R = UTPM.qr(A)


# print UTPM.triu(R) - R

# assert_array_almost_equal(UTPM.triu(R).data,  R.data)

# # for d in range(D):
# #     for p in range(P):
# #         print (R.data[d,p] - numpy.triu(R.data[d,p])).argmax()
# #         # assert_array_almost_equal(R.data[d,p], numpy.triu(R.data[d,p]))

# # assert_array_almost_equal(A.data, UTPM.dot(Q,R).data)
# # assert_array_almost_equal(0, (UTPM.dot(Q.T,Q) - numpy.eye(M)).data)

# # Q2 = Q[:,N:]
# # # assert_array_almost_equal(0, UTPM.dot(A.T, Q2).data)

# # print UTPM.dot(A.T, Q2).data.max()

    
    
    
