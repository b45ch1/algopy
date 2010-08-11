from numpy import array, log, exp
from algopy import UTPM

x = UTPM(numpy.zeros((2,1,3)))
x.data[0,0] = [3,5,7]
x.data[1,0] = [1,0,0]

A = UTPM(numpy.zeros((2,1,3,2)))
A[0,0] = numpy.sin(x[0])**2 + x[1]
A[0,1] = x[0]
A[1,0] = numpy.exp(x[0]/x[1])
A[1,1] = x[2]
A[2,0] = numpy.log(x[0] + x[2]*x[1])

print 'A =', A

y = eigh(inv(dot(A.T, A)))[0][-1]

print 'Phi(x) = ', y.data[0]
print 'd/dx_1 Phi(x) = ', y.data[1] 

