import numpy; from numpy import sin,cos
from algopy import UTPM

def f(x):
    return sin(cos(x) + sin(x))

D = 100; P = 1
x = UTPM(numpy.zeros((D,P)))
x.data[0,0] = 0.3
x.data[1,0] = 1

y = f(x)
print('coefficients of y =', y.data[:,0])

import matplotlib.pyplot as pyplot; import os
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
pyplot.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)),'taylor_approximation.png'))
# pyplot.show()


