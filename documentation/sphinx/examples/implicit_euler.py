import numpy; from numpy import sin,cos; from algopy import UTPM, zeros
D,P = 2,1
x = UTPM(numpy.zeros((D,P,2)))
x.data[0,:,0] = 1
p = UTPM(numpy.zeros((D,P)))
p.data[0,:] = 3; p.data[1,:] = 1

def f(t, x, p):
    retval = x.copy()
    retval[0] = x[1]
    retval[1] = -p* x[0]
    return retval
    
def implicit_euler(f_fcn, x0, ts, p):
    """ implicit euler with fixed stepsizes, using Newton's method to solve
    the occuring implicit system of nonlinear equations
    """
    
    def F_fcn(x_new, x, t_new, t, p):
        """ implicit function to solve:  0 = F(x_new, x, t_new, t_old)"""
        return (t_new - t) * f_fcn(t_new, x_new, p) - x_new + x
        
        
   
    x = x0.copy()
    D,P,N = x.data.shape
    y = UTPM(numpy.zeros((D,N,N)))
    x_new = x.copy()

    x_list = [x.data.copy() ]
    for nts in range(ts.size-1):
        h = ts[nts+1] - ts[nts]
        
        # compute the Jacobian
        y.data[0,:]   = x.data[0,:]
        y.data[1,:,:] = numpy.eye(N)
        F = F_fcn(y, x.data[0,0], ts[nts+1], ts[nts], p.data[0,0])
        J = F.data[1,:,:].T

        # d=0: apply Newton's method to solve 0 = F_fcn(x_new, x, t_new, t)
        x_new[...] = x[...]
        rel_error = numpy.inf
        while rel_error > 10**-6:
            delta_x = numpy.linalg.solve(J, F_fcn(x_new.data[0,0], x.data[0,0], ts[nts+1], ts[nts], p.data[0,0]))
            x_new.data[0,0] -= delta_x
            rel_error = numpy.linalg.norm(delta_x)
        
        # # d>0: compute higher order coefficients
        # for d in range(1,D):
        #     x_new.data[d,0] = numpy.linalg.solve
        
        x[...] = x_new[...]
        x_list.append(x.data.copy())
        
    return numpy.array(x_list)
    

# compute AD solution
ts = numpy.linspace(0,2*numpy.pi,1000)
xs = implicit_euler(f, x, ts, p)

print xs
# analytical solution
def x_analytical(t,p):
    return numpy.cos(numpy.sqrt(p)*t)
    
def x_p_analytical(t,p):
    return -0.5*numpy.sin(numpy.sqrt(p)*t)*p**(-0.5)*t

print xs.shape
import matplotlib.pyplot as pyplot
pyplot.plot(ts, xs[:,0,0,0], ',k-', label = r'$x(t)$')
pyplot.plot(ts, x_analytical(ts,p.data[0,0]), 'k-.', label = r'analytic $x(t)$')
pyplot.plot(ts, xs[:,1,0,0], ',r-', label = r'$x_p(t)$')
pyplot.plot(ts, x_p_analytical(ts,p.data[0,0]), 'r-.', label = r'analytic $x_p(t)$')

pyplot.xlabel('time $t$')
pyplot.legend(loc='best')
pyplot.grid()
pyplot.savefig('implicit_euler.png')
pyplot.show()
     
