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

print 'y0 = ', y0
print 'y  = ', y
print 'y.shape =', y.shape
print 'y.data.shape =', y.data.shape
print 'dF/dx(x0) * x1 =', y.data[1,0]
