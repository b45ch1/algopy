import numpy; from numpy import sin,cos; from taylorpoly import UTPS
x = numpy.array([UTPS([1,0,0],P=2), UTPS([0,0,0],P=2)])
p = UTPS([3,1,0],P=2)
def f(x):
    return numpy.array([x[1],-p * x[0]])

ts = numpy.linspace(0,2*numpy.pi,100)
x_list = [[xi.data.copy() for xi in x]]
for nts in range(ts.size-1):
    h = ts[nts+1] - ts[nts]
    x = x + h * f(x)
    x_list.append([xi.data.copy() for xi in x])

xs = numpy.array(x_list)
import matplotlib.pyplot as pyplot
pyplot.plot(ts, xs[:,0,0], '.k-', label = r'$x(t)$')
pyplot.plot(ts, xs[:,0,1], '.r-', label = r'$x_p(t)$')
pyplot.xlabel('time $t$')
pyplot.legend(loc='best')
pyplot.grid()
pyplot.show()
    
    
    
