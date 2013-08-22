import numpy; from numpy import sin,cos; from algopy import UTPM, zeros
D,P = 4,1
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
        
    def J_fcn(x_new, x, t_new, t, p):
        """ computes the Jacobian of F_fcn
        all inputs are double arrays
        """
        y = UTPM(numpy.zeros((D,N,N)))
        y.data[0,:]   = x_new
        y.data[1,:,:] = numpy.eye(N)
        F = F_fcn(y, x, t_new, t, p)
        return F.data[1,:,:].T
        
        
    x = x0.copy()
    D,P,N = x.data.shape
    x_new = x.copy()

    x_list = [x.data.copy() ]
    for nts in range(ts.size-1):
        h = ts[nts+1] - ts[nts]
        x_new.data[0,...] = x.data[0,...]
        x_new.data[1:,...] = 0

        # compute the Jacobian at x
        J = J_fcn(x_new.data[0,0], x.data[0,0], ts[nts+1], ts[nts], p.data[0,0])

        # d=0: apply Newton's method to solve 0 = F_fcn(x_new, x, t_new, t)
        step = numpy.inf
        while step > 10**-10:
            delta_x = numpy.linalg.solve(J, F_fcn(x_new.data[0,0], x.data[0,0], ts[nts+1], ts[nts], p.data[0,0]))
            x_new.data[0,0] -= delta_x
            step = numpy.linalg.norm(delta_x)
        
        # d>0: compute higher order coefficients
        J = J_fcn(x_new.data[0,0], x.data[0,0], ts[nts+1], ts[nts], p.data[0,0])
        for d in range(1,D):
            F = F_fcn(x_new, x, ts[nts+1], ts[nts], p)
            x_new.data[d,0] = -numpy.linalg.solve(J, F.data[d,0])
        
        x.data[...] = x_new.data[...]
        x_list.append(x.data.copy())
        
    return numpy.array(x_list)
    

# compute AD solution
ts = numpy.linspace(0,2*numpy.pi,1000)
xs = implicit_euler(f, x, ts, p)

# print xs
# analytical solution
def x_analytical(t,p):
    return numpy.cos(numpy.sqrt(p)*t)
    
def x_p_analytical(t,p):
    return -0.5*numpy.sin(numpy.sqrt(p)*t)*p**(-0.5)*t

print(xs.shape)
import matplotlib.pyplot as pyplot; import os
pyplot.plot(ts, xs[:,0,0,0], ',k-', label = r'$x(t)$')
pyplot.plot(ts, x_analytical(ts,p.data[0,0]), 'k-.', label = r'analytic $x(t)$')
pyplot.plot(ts, xs[:,1,0,0], ',r-', label = r'$x_p(t)$')
pyplot.plot(ts, x_p_analytical(ts,p.data[0,0]), 'r-.', label = r'analytic $x_p(t)$')
pyplot.plot(ts, xs[:,2,0,0], ',b-', label = r'$x_{pp}(t)$')
pyplot.plot(ts, xs[:,3,0,0], ',m-', label = r'$x_{ppp}(t)$')

pyplot.title('analytical and implicit Euler solution')
pyplot.xlabel('time $t$')
pyplot.legend(loc='best')
pyplot.grid()
pyplot.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)),'implicit_euler.png'))
# pyplot.show()
     
