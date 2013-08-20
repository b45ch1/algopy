import numpy; from numpy import sin,cos; from algopy import UTPM, zeros
D,P = 4,1
x = UTPM(numpy.zeros((D,P,2)))
x.data[0,:,0] = 1
p = UTPM(numpy.zeros((D,P)))
p.data[0,:] = 3; p.data[1,:] = 1

def f(x):
    retval = x.zeros_like()
    retval[0] = x[1]
    retval[1] = -p* x[0]
    return retval

# compute AD solution
ts = numpy.linspace(0,2*numpy.pi,2000)
x_list = [x.data.copy() ]
for nts in range(ts.size-1):
    h = ts[nts+1] - ts[nts]
    x = x + h * f(x)
    x_list.append(x.data.copy())

# analytical solution
def x_analytical(t,p):
    return numpy.cos(numpy.sqrt(p)*t)
    
def x_p_analytical(t,p):
    return -0.5*numpy.sin(numpy.sqrt(p)*t)*p**(-0.5)*t

xs = numpy.array(x_list)
print(xs.shape)
import matplotlib.pyplot as pyplot; import os
pyplot.plot(ts, xs[:,0,0,0], ',k-', label = r'$x(t)$')
pyplot.plot(ts, x_analytical(ts,p.data[0,0]), 'k-.', label = r'analytic $x(t)$')
pyplot.plot(ts, xs[:,1,0,0], ',r-', label = r'$x_p(t)$')
pyplot.plot(ts, x_p_analytical(ts,p.data[0,0]), 'r-.', label = r'analytic $x_p(t)$')
pyplot.plot(ts, xs[:,2,0,0], ',b-', label = r'$x_{pp}(t)$')
pyplot.plot(ts, xs[:,3,0,0], ',m-', label = r'$x_{ppp}(t)$')



pyplot.xlabel('time $t$')
pyplot.legend(loc='best')
pyplot.grid()
pyplot.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)),'explicit_euler.png'))
# pyplot.show()
    
