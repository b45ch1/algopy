#!/usr/bin/env python
import numpy; npy = numpy #alias
import numpy.linalg as lin
import scipy.integrate
import scipy as spy


from prettyplotting import *
from forward_mode import *

#const parameters that define the problem
GRVTY_CONST = 9.81
INITIAL_SPEED = 1600.
C_W = 0.15
DENSITY_OF_AIR = 1.3
MASS_OF_PROJECTILE =  194.
AREA_OF_PROJECTILE = (0.5 *0.21)**2 * npy.pi
TARGET_POINT = npy.array([40000.,0.])

#parameters for the ODE solver
start=0.
end=1.
numsteps=200
time=npy.linspace(start,end,numsteps)

#parameters for Newton's method
break_threshold = 0.0001
num_its = 0
add_num_its = 10
theta = 0.4 * npy.pi
T = 100.

#output variables
bs = [] #distance between the bullet at the end time T and the target point
thetas_and_Ts = []



def g(z,t, T):
	"""Vector field of the ODE d/dt z = g(z,t,T). More specifically, the ODE is the equations of motion that governs how a projectile moves, given initial conditions."""
	return npy.array([
			z[2]*T,
			z[3]*T,
			(0. - 0.5/MASS_OF_PROJECTILE * AREA_OF_PROJECTILE * DENSITY_OF_AIR * C_W * npy.sqrt(z[2]**2 + z[3]**2)*z[2])*T,
			(-GRVTY_CONST - 0.5/MASS_OF_PROJECTILE * AREA_OF_PROJECTILE * DENSITY_OF_AIR * C_W * npy.sqrt(z[2]**2 + z[3]**2)*z[3])*T
		])
		

def foo(*args,**keywords):
  print "Number of arguments:", len(args)
  print "Arguments are: ", args
  print "Number of keywords: ", len(keywords)
  print "Keyword passed arguments are: ", keywords

foo(3,2, [2,3], lala = npy.array([1,2,3]))

#gradient(g,[1,2,3,4])

#def G(v,t,T):
	#"""Vector field of the ODE d/dt v = G(v,t,T),
	 #where v = [d/d(theta) z_1, ..., d/d(theta)z_4, d/dT z1, ..., d/dT z4, z_1,...,z_4].
	#Solving this ODE yields d/d(theta)z(t=1) and d/dT z(t=1) which is needed in Newton's method."""
	
	#epsilon = 0.001
	#rv = v.copy() #return value

	
	#rv[:4] =  (g(v[8:]+epsilon*v[:4],t,T) - g(v[8:],t,T))/epsilon
	#rv[4:8] =  (g(v[8:]+epsilon*v[4:8],t,T) - g(v[8:],t,T))/epsilon + (g(v[8:],t,T+epsilon) - g(v[8:],t,T))/epsilon
	#rv[8:] = g(v[8:],t,T)
	#return rv

##plot trajectories
#pyl.figure()
#pyl.plot([TARGET_POINT[0]],[TARGET_POINT[1]],'ro') #target

#delta = 12345
#while lin.norm(delta)>break_threshold:
	#print 'theta=',theta,'T=',T
	#num_its +=1
	#thetas_and_Ts.append([theta,T])

	##compute initial values
	#v = npy.zeros(12,dtype=float)
	#v[10:12] = [npy.cos(theta) * INITIAL_SPEED, npy.sin(theta) * INITIAL_SPEED]
	#v[2:4] = [-npy.sin(theta) * INITIAL_SPEED, npy.cos(theta) * INITIAL_SPEED]
	##solve ODE
	#y = scipy.integrate.odeint(G,v,time,args=(T,))
	
	##Newton Iteration
	#A = npy.zeros((2,2),dtype=float); A[:,0] = y[-1,:2]; A[:,1] = y[-1,4:6]
	#b = TARGET_POINT - npy.array([y[-1,8],y[-1,9]])
	#delta = lin.solve(A,b)
	#theta += delta[0];	T += delta[1]
	#bs.append(lin.norm(b))
	#pyl.plot(y[:,8],y[:,9])

##do more iterations to find good approximation to (\theta^*, \T^*) (needed for the order of convergence)
#for k in range(add_num_its):
	##compute initial values
	#v = npy.zeros(12,dtype=float)
	#v[10:12] = [npy.cos(theta) * INITIAL_SPEED, npy.sin(theta) * INITIAL_SPEED]
	#v[2:4] = [-npy.sin(theta) * INITIAL_SPEED, npy.cos(theta) * INITIAL_SPEED]
	##solve ODE
	#y = scipy.integrate.odeint(G,v,time,args=(T,))
	
	##Newton Iteration
	#A = npy.zeros((2,2),dtype=float); A[:,0] = y[-1,:2]; A[:,1] = y[-1,4:6]
	#b = TARGET_POINT - npy.array([y[-1,8],y[-1,9]])
	#delta = lin.solve(A,b)
	#theta += delta[0];	T += delta[1]

#pyl.grid(); pyl.xlabel(r'$x$ [m]'); pyl.ylabel(r'$y$ [m]'); ymin, ymax = pyl.ylim(); pyl.ylim(-100.,ymax)
#pyl.title("trajectories found with Newton's method")
#pyl.savefig('single_shooting_newtons_method.eps')

###comparing AD solution to finite differences approximation of the inverse Jacobian A
##fd_error = []
##for epsilon in [10**(-i) for i in range(1,20)]:
	##w = [0,0,npy.cos(theta) * INITIAL_SPEED, npy.sin(theta) * INITIAL_SPEED]
	##w_epsilon = [0,0,npy.cos(theta+epsilon) * INITIAL_SPEED, npy.sin(theta+epsilon) * INITIAL_SPEED]
	##y = scipy.integrate.odeint(g,w,time,args=(T,))
	##y_epsilon_theta = scipy.integrate.odeint(g,w_epsilon,time,args=(T,))
	##y_epsilon_T = scipy.integrate.odeint(g,w,time,args=(T+epsilon,))
	##C = npy.zeros((2,2),dtype=float)
	##C[:,0] = (y_epsilon_theta[-1,:2]- y[-1,:2])/epsilon
	##C[:,1] = (y_epsilon_T[-1,:2]- y[-1,:2])/epsilon
	###b = TARGET_POINT - npy.array([y[-1,0],y[-1,1]])
	###delta2 = lin.solve(C,b)
	##fd_error.append( lin.norm(C-A) )
##pyl.figure()
##pyl.plot(fd_error,'r.')

##convergence of Newton's method
#pyl.figure()
#thetas_and_Ts = npy.asarray(thetas_and_Ts)
#thetas_and_Ts_transformed = thetas_and_Ts.copy()
#thetas_and_Ts_transformed[:,0]-=theta #substract last and therefore best estimate of theta
#thetas_and_Ts_transformed[:,1]-=T
#thetas_and_Ts_transformed_log_norm = npy.asarray([npy.log(lin.norm(thetas_and_Ts_transformed[i,:]))/npy.log(10) for i in range(num_its)])

##solve least squares problem to fit a nonlinear model

#A = npy.zeros((num_its-1,2),dtype=float)
#A[:,0] = 1.
##print npy.shape(thetas_and_Ts_transformed_log_norm[:-1])
##print npy.shape(A)

#A[:,1] = thetas_and_Ts_transformed_log_norm[:-1]
##print A
#B = npy.dot(A.T,A)
##print B
#b = thetas_and_Ts_transformed_log_norm[1:]

##print 'b=',b
#c = npy.dot(A.T,b)
##print 'c=',c
#x = lin.solve( B,c)
##print 'x=',x
#valplot = pyl.plot(range(num_its), thetas_and_Ts_transformed_log_norm, 'r.')
##plot fitted model
#z = npy.zeros(num_its)
#z[0] = thetas_and_Ts_transformed_log_norm[0]
#for k in range(1,num_its):
	#z[k] = x[0] + x[1] * z[k-1]
#regplot = pyl.plot(range(num_its),z)
#pyl.title(r'Error convergence $\|x_{n+1} - x^* \| \leq C \| x_n - x^* \|^{p=%0.3f}$'%x[1])
#pyl.legend((regplot,valplot),('fitted model','measured data points'), loc = 3)
#pyl.xlabel(r'iteration $n$ [ ]')
#pyl.ylabel(r' $\log_{10}( \| x_n - x_N \|,\; n \leq N$')
#pyl.savefig('order_of_convergence.eps')

##convergence of the residuals
#pyl.figure()
#its = npy.shape(bs)[0]
#log_bs = npy.log(bs)/npy.log(10)

##make regression
#reg_coeffs = spy.polyfit(npy.arange(its),log_bs,1, full=False)
#regplot = pyl.plot(npy.linspace(0,its-1,100), spy.polyval(reg_coeffs, npy.linspace(0,its-1,100)))
#valplot = pyl.plot(npy.arange(its),log_bs,'r.')
#pyl.legend((regplot,valplot),('fitted model','measured data points'))
#pyl.xlabel('iteration [ ]')
#pyl.ylabel(r'error $\log_{10} \| x(1;(\theta,T) - x_{\rm target} \|_2$')
#pyl.title(r'Linear model of residuals $r(x) = %0.2f x +  %0.2f$'%(reg_coeffs[0],reg_coeffs[1]))
#pyl.savefig('residuals.eps')

#pyl.show()

