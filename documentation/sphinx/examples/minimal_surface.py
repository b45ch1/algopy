import numpy
import algopy

def O_tilde(u):
    """ this is the objective function"""
    M = numpy.shape(u)[0]
    h = 1./(M-1)
    return M**2*h**2 + numpy.sum(0.25*( (u[1:,1:] - u[0:-1,0:-1])**2 + (u[1:,0:-1] - u[0:-1, 1:])**2))


def dO_tilde(u):
    g = numpy.zeros(numpy.shape(u))
    g[1:-1, 1:-1] = 2 * u[1:-1,1:-1] - 0.5*( u[0:-2,0:-2]  + u[2:,2:]  + u[:-2, 2:] + u [2:, :-2] )
    return g

# INITIAL VALUES
M = 5
h = 1./M
u = numpy.zeros((M,M),dtype=float)
u[0,:]=  [numpy.sin(numpy.pi*j*h/2.) for j in range(M)]
u[-1,:] = [ numpy.exp(numpy.pi/2) * numpy.sin(numpy.pi * j * h / 2.) for j in range(M)]
u[:,0]= 0
u[:,-1]= [ numpy.exp(i*h*numpy.pi/2.) for i in range(M)]

u = algopy.UTPM(u.reshape((1,1) + u.shape))

cg = algopy.CGraph()
Fu = algopy.Function(u)
Fy = O_tilde(Fu)
cg.trace_off()
cg.independentFunctionList = [Fu]
cg.dependentFunctionList = [Fy]

g = cg.gradient([u.data[0,0]])[0]

# cg.pullback([algopy.UTPM(numpy.ones((1,1)))])
# g = u.xbar.data[0,0]

# on the edge the analytical solution is fixed to zero
g[:,0]  = 0
g[0,:]  = 0
g[:,-1] = 0
g[-1,:] = 0

print g - dO_tilde(u.data[0,0])

# def test_solve_minimal_surface_optimization_problem_with_projected_gradients(self):
#     """
#     This is a minimal surface problem, discretized on a regular mesh with box constraints using projected gradients.
#     The necessary gradient is computed with adolc.
#     """

#     def projected_gradients(x0, ffcn,dffcn, box_constraints, beta = 0.5, delta = 10**-3, epsilon = 10**-2, max_iter = 1000, line_search_max_iter = 100):
#         """
#         INPUT:	box_constraints		[L,U], where L (resp. U) vector or matrix with the lower (resp. upper) bounds
#         """
#         x = x0.copy()
#         L = numpy.array(box_constraints[0])
#         U = numpy.array(box_constraints[1])
#         def pgn(s):
#             a = 1.* (x>L)
#             b = 1.*(abs(x-L) <0.00001)
#             c = 1.*(s>0)
#             d = numpy.where( a + (b*c))
#             return numpy.sum(s[d])

#         def P(x, s, alpha):
#             x_alpha = x + alpha * s
#             a = x_alpha-L
#             b = U - x_alpha
#             return x_alpha - 1.*(a<0) * a + b * 1. * (b<0)

            
#         s = - dffcn(x)
#         k = 0
#         while pgn(s)>epsilon and k<= max_iter:
#             k +=1
#             s = - dffcn(x)
#             for m in range(line_search_max_iter):
#                 #print 'm=',m
#                 alpha = beta**m
#                 x_alpha = P(x,s,alpha)
#                 if ffcn( x_alpha ) - ffcn(x) <= - delta * numpy.sum(s* (x_alpha - x)):
#                     break
#             x_old = x.copy()
#             x = x_alpha

#         return x_old,s


    

#     def O_tilde(u):
#         """ this is the objective function"""
#         M = numpy.shape(u)[0]
#         h = 1./(M-1)
#         return M**2*h**2 + numpy.sum(0.25*( (u[1:,1:] - u[0:-1,0:-1])**2 + (u[1:,0:-1] - u[0:-1, 1:])**2))

#     # INITIAL VALUES
#     M = 20
#     h = 1./M
#     u = numpy.zeros((M,M),dtype=float)
#     u[0,:]=  [numpy.sin(numpy.pi*j*h/2.) for j in range(M)]
#     u[-1,:] = [ numpy.exp(numpy.pi/2) * numpy.sin(numpy.pi * j * h / 2.) for j in range(M)]
#     u[:,0]= 0
#     u[:,-1]= [ numpy.exp(i*h*numpy.pi/2.) for i in range(M)]

#     # tape the function evaluation
#     trace_on(1)
#     au = adouble(u)
#     independent(au)
#     ay = O_tilde(au)
#     dependent(ay)
#     trace_off()


#     def dO_tilde(u):
#         ru = numpy.ravel(u)
#         rg = gradient(1,ru)
#         g = numpy.reshape(rg, numpy.shape(u))
        
#         # on the edge the analytical solution is fixed to zero
#         g[:,0]  = 0
#         g[0,:]  = 0
#         g[:,-1] = 0
#         g[-1,:] = 0
        
#         return g

    
#     # X AND Y PARTITION
#     x_grid = numpy.linspace(0,1,M)
#     y_grid = numpy.linspace(0,1,M)

#     # BOX CONSTRAINTS
#     lo = 2.5
#     L = numpy.zeros((M,M),dtype=float)

#     for n in range(M):
#         for m in range(M):
#             L[n,m] = 2.5 * ( (x_grid[n]-0.5)**2 + (y_grid[m]-0.5)**2 <= 1./16)

#     U = 100*numpy.ones((M,M),dtype=float)

#     Z,s = projected_gradients(u,O_tilde,dO_tilde,[L,U])


#     # THIS CODE BELOW ONLY WORKS FOR MATPLOTLIB 0.91X
#     try:
#         import pylab
#         import matplotlib.axes3d as p3

#         x = y = range(numpy.shape(Z)[0])
#         X,Y = numpy.meshgrid(x_grid,y_grid)

#         fig=pylab.figure()
#         ax = p3.Axes3D(fig)
#         ax.plot_wireframe(X,Y,Z)

#         xs = Z + s
#         for n in range(M):
#             for m in range(M):
#                 ax.plot3d([x_grid[m],x_grid[m]], [ y_grid[n], y_grid[n]], [Z[n,m], xs[n,m]], 'r')
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         pylab.title('Minimal Surface')
#         pylab.savefig('./3D_plot.png')
#         pylab.savefig('./3D_plot.eps')

#         #pylab.show()
#     except:
#         print '3d plotting with matplotlib failed'
#         pass

#     # Plot with MAYAVI
#     try:
#         import enthought.mayavi.mlab as mlab
#         mlab.figure()
#         mlab.view(azimuth=130)
#         s = mlab.surf(x, y, Z, representation='wireframe', warp_scale='auto', line_width=1.)
#         mlab.savefig('./mayavi_3D_plot.png')
#         #mlab.show()

#     except:
#         pass
