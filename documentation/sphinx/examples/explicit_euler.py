import numpy; from numpy import sin,cos; from taylorpoly import UTPS
x = numpy.array([UTPS([1,0,0],P=2), UTPS([0,0,0],P=2)])
p = UTPS([3,1,0],P=2)
def f(x):
    return numpy.array([x[1],-p * x[0]])

# compute AD solution
ts = numpy.linspace(0,2*numpy.pi,3000)
x_list = [[xi.data.copy() for xi in x]]
for nts in range(ts.size-1):
    h = ts[nts+1] - ts[nts]
    x = x + h * f(x)
    x_list.append([xi.data.copy() for xi in x])
    
# analytical solution
def x_analytical(t,p):
    return numpy.cos(numpy.sqrt(p)*t)
    
def x_p_analytical(t,p):
    return -0.5*numpy.sin(numpy.sqrt(p)*t)*p**(-0.5)*t

xs = numpy.array(x_list)
import matplotlib.pyplot as pyplot
pyplot.plot(ts, xs[:,0,0], ',k-', label = r'$x(t)$')
pyplot.plot(ts, x_analytical(ts,p.data[0]), 'k-.', label = r'analytic $x(t)$')
pyplot.plot(ts, xs[:,0,1], ',r-', label = r'$x_p(t)$')
pyplot.plot(ts, x_p_analytical(ts,p.data[0]), 'r-.', label = r'analytic $x_p(t)$')

pyplot.xlabel('time $t$')
pyplot.legend(loc='best')
pyplot.grid()
pyplot.savefig('explicit_euler.png')
pyplot.show()
    
