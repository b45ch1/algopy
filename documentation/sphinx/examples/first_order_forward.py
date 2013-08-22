import numpy; from numpy import array
from algopy import UTPM, zeros

def F(x):
    y = zeros(3, dtype=x)
    y[0] = x[0]*x[1]
    y[1] = x[1]*x[2]
    y[2] = x[2]*x[0]
    return y

x0 = array([1,3,5],dtype=float)
x1 = array([1,0,0],dtype=float)

D = 2; P = 1
x = UTPM(numpy.zeros((D,P,3)))
x.data[0,0,:] = x0
x.data[1,0,:] = x1

# normal function evaluation
y0 = F(x0)

# UTP function evaluation
y = F(x)

print('y0 = ', y0)
print('y  = ', y)
print('y.shape =', y.shape)
print('y.data.shape =', y.data.shape)
print('dF/dx(x0) * x1 =', y.data[1,0])



import numpy; from numpy import log, exp, sin, cos, abs
import algopy; from algopy import UTPM, dot, inv, zeros

def f(x):
    A = zeros((2,2),dtype=x)
    A[0,0] = numpy.log(x[0]*x[1])
    A[0,1] = numpy.log(x[1]) + exp(x[0])
    A[1,0] = sin(x[0])**2 + abs(cos(x[0]))**3.1
    A[1,1] = x[0]**cos(x[1])
    return log( dot(x.T,  dot( inv(A), x)))

x = numpy.array([3.,7.])
x = UTPM.init_jacobian(x)

y = f(x)

print('normal function evaluation f(x) = ',y.data[0,0])
print('Jacobian df/dx = ', UTPM.extract_jacobian(y))

