import numpy; from numpy import sin,cos
from algopy import UTPM

def f(x):
    return sin(cos(x) + sin(x))

D = 100; P = 1
x = UTPM(numpy.zeros((D,P)))
x.data[0,0] = 0.3
x.data[1,0] = 1

y = f(x)
print 'coefficients of y =', y.data[:,0]

import matplotlib.pyplot as pyplot
zs = numpy.linspace(-1,2,100)
ts = zs -0.3
fzs = f(zs)

for d in [2,4,10,50,100]:
    yzs = numpy.polyval(y.data[:d,0][::-1],ts)
    pyplot.plot(zs,yzs, label='%d\'th order approx.'%(d-1))
    
pyplot.plot([0.3], f([0.3]), 'ro')
pyplot.plot(zs,fzs, 'k.')
pyplot.grid()
pyplot.legend(loc='best')
pyplot.xlabel('x')
pyplot.ylabel('y')
pyplot.savefig('taylor_approximation.png')
# pyplot.show()



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
