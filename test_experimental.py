from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utpm import *
from algopy.globalfuncs import *



# D,P,M,N = 3,1,4,2

# x = UTPM(numpy.zeros((D,P,M,N)))
# y = UTPM(numpy.zeros((D,P)))

# x,y = UTPM.postpend_ones(x,y)

# x + y

D,P,M,N = 3,1,3,2
A = UTPM(numpy.zeros((D,P,M,M)))


x = numpy.random.rand(3,2)
A.data[0,0] = numpy.dot(x,x.T)
x = numpy.random.rand(3,2)
A.data[1,0] = numpy.dot(x,x.T)


         
# A.data[1,0,:N,:] = 2.
# A.data[1,0] = 1.
# A.data[2,0] = 2


print 'A=',A

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

    
    
    
