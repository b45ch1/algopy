#!/usr/bin/env python
from pylab import *
import sys
sys.path = ['..'] + sys.path
from matrix_ad import *
import adolc
import numpy.random
import scipy.optimize

from prettyplotting import * # comment this out if not available


"""
# We look at the following OED problem:
# variables:
#        p = v[:Np] = parameters, Np is number of parameters
#        q = v[Np:] = control variables
#
# model: ODE
#        dx/dt = f(t,x,p) = p0
#        x(0)  = x_0 = p1
# measurement model:
#        h(t,x,v) = x(t,v)
# parameter estimation:
#        F(p) = [F1, ..., FM].T
#        F = Sigma^-1 ( eta - h(t,x,p,q) )
#        eta are the measurements
#        p_   = argmin_p |F(p)|_2^2
# OED:
#        J = dF/dp
#        C = (J^T J)^-1  covariance matrix of the parameters
#        Phi(q) = tr(C(q))
#        q_ = argmin_q Phi(q)
"""
def explicit_euler(x0,f,ts,p,q):
	N = size(ts)
	if isinstance(p[0],adolc.adouble):
		x = array([adolc.adouble(0) for m in range(Nm)])
	else:
		x = zeros(N)
	x[0] = x0
	for n in range(1,N):
		h = ts[n] - ts[n-1]
		x[n]= x[n-1] + h*f(ts[n-1],x[n-1],p,q)
	return x

def measurement_model(x,p,q):
	return x

def f(t,x,p,q):
	""" rhs of the ODE"""
	return p[1] + q[0]*x

def F(p,q,ts,Sigma, etas):
	x = explicit_euler(p[0],f,ts,p,q)
	h = measurement_model(x,p,q)
	return dot(Sigma, h-etas)
	
def Phi(J):
	""" prototypical OED objective function"""
	return trace(inv(dot(J.T,J)))

if __name__ == "__main__":

	# problem setup
	Nm = 100    # number of measurements
	Np = 2      # number of parameters
	Nq = 1      # number of control variables
	Nv = Np + Nq
	ts = linspace(0,10,Nm)
	Sigma = eye(Nm)
	p = array([10.,2.])
	q = array([-1.])
	v = concatenate((p,q))

	# generate pseudoe measurement data
	p[0]+=3.; 	p[1] += 2.
	x = explicit_euler(p[0],f,ts,p,q)
	h = measurement_model(x,p,q)
	etas = h + numpy.random.normal(size=Nm)
	p[0]-= 3.;	p[1] -= 2.

	# taping F
	av = array([adolc.adouble(0) for i in range(Nv)])
	y = zeros(Nm)
	adolc.trace_on(1)
	av[0].is_independent(p[0])
	av[1].is_independent(p[1])
	av[2].is_independent(q[0])
	ay = F(av[:Np],av[Np:],ts,Sigma,etas)
	for m in range(Nm):
		y[m] = adolc.depends_on(ay[m])
	adolc.trace_off()

	# taping measurement_model
	av = array([adolc.adouble(0) for i in range(Nv)])
	y = zeros(Nm)
	adolc.trace_on(2)
	av[0].is_independent(p[0])
	av[1].is_independent(p[1])
	av[2].is_independent(q[0])
	ax = explicit_euler(av[0],f,ts,av[:Np],av[Np:])
	ay = measurement_model(ax, av[:Np],av[Np:])
	for m in range(Nm):
		y[m] = adolc.depends_on(ay[m])
	adolc.trace_off()

	# PERFORM PARAMETER ESTIMATION
	def dFdp(p,q,ts,Sigma, etas):
		v[:Np] = p[:]
		return adolc.jacobian(1,v)[:,:Np]
	res = scipy.optimize.leastsq(F,p,args=(q,ts,Sigma,etas), Dfun = dFdp, full_output = True)

	# plotting solution of parameter estimation and starting point
	p[0]+=3;	p[1] += 2.
	x = explicit_euler(p[0],f,ts,p,q)
	y = adolc.zos_forward(2,v,0)
	p[0]-= 3;	p[1] -= 2.
	starting_plot = plot(ts,x)
	x = explicit_euler(p[0],f,ts,p,q)
	correct_plot = plot(ts,x,'b.')
	meas_plot = plot(ts,etas,'r.')
	x = explicit_euler(res[0][0],f,ts,res[0],q)
	est_plot = plot(ts,x)
	plot(ts,etas,'r.')
	
	hplot = plot(ts,y,'g.')
	
	xlabel(r'time $t$ []')
	ylabel(r'measurement function $h(t,x,p,q)$')
	legend((meas_plot,starting_plot,correct_plot,est_plot,hplot),('measurements','initial guess','true','estimated','measurement model'))
	title('Parameter Estimation')
	savefig('parameter_estimation.png')
	savefig('parameter_estimation.eps')



	# PERFORM OED
	# ===========
	v0 = v.copy()

	# tape the objective function with Algopy
	J=adolc.jacobian(2,v)[:,:2]

	cg = CGraph()
	J0 = J.copy()
	J1 = zeros(shape(J))
	FJ = Function(Mtc(J0,J1))
	Ff = Phi(FJ)
	cg.independentFunctionList = [FJ]
	cg.dependentFunctionList = [Ff]
	cg.plot('testgraph.png')
	cg.plot('testgraph.svg')
	
	# perform steepest descent optimization
	vbar = inf
	count = 0
	while numpy.linalg.norm(vbar)>10**-8:
		count +=1
		# 1: evaluation of J
		Jtc=Mtc(adolc.jacobian(1,v)[:,:Np],J1)


		# 2: forward evaluation of Phi
		cg.forward([Jtc])
		#print 'Phi=',cg.dependentFunctionList[0].x.X
	
		# 3: reverse evaluation of Phi
		cg.reverse([Mtc([[1.]],[[0.]])])
		Jbar = FJ.xbar.X
		#print 'Jbar=',Jbar

		# 4: reverse evaluation of J
		x = v
		D = 1
		keep = D+1
		V = zeros((Nv,D))
		vbar = zeros(Nv)
		for np in range(Np):
			V[np,0] = 1
			u = (Jbar.T)[np,:].copy()
			adolc.hos_forward(1,x,V,keep)
			Z = adolc.hos_reverse(1,D,u)
			V[np,0] = 0
			print 'Z=',Z
			vbar += Z[2,1]
		print vbar
		exit()
		#update v:  x_k+1 = v_k - g
		v[2:] -= vbar[2:]
	print 'number of iterations =',count
	print 'v_opt =',v
	print 'v0=',v0

	# plot Phi
	# --------
	def dFdp(p,q,ts,Sigma, etas):
		v = concatenate((p,q))
		return adolc.jacobian(2,v)[:,:Np]
	
	qs = linspace(-1,2,100)
	Phis = []
	for q in qs:
		q = array([q])
		J = dFdp(p,q,ts,Sigma, etas)
		Phis.append(Phi(J))

	figure()
	qplot = plot(qs,Phis,'k')
	optimal_plot = plot([v[2]], cg.dependentFunctionList[0].x.X[0] , 'go')
	xlabel(r'$q$')
	ylabel(r'$\Phi(q)$')
	legend((qplot, optimal_plot), (r'$\Phi(q)$','computed optimal solution'))
	title('Optimal Design of Experiments')
	savefig('odoe_objective_function.png')
	savefig('odoe_objective_function.eps')

	# plot state for initial value and optimal value
	# ----------------------------------
	figure()
	p0 = v0[:2]
	q0 = array([v0[2]])
	x0 = explicit_euler(p0[0],f,ts,p0,q0)
	initial_plot = plot(ts,x0,'b')

	p_opt = v[:2]
	q_opt = array([v[2]])
	x_opt = explicit_euler(p_opt[0],f,ts,p_opt,q_opt)
	opt_plot = semilogy(ts,x_opt,'r')
	xlabel(r'time $t$ [sec]')
	ylabel(r'$x(t)$')
	legend((opt_plot,initial_plot),('optimal state traj.', 'initial state traj.'))
	
	#show()
	
	










