#!/usr/bin/env python
from pylab import *
import sys
sys.path = ['..'] + sys.path
from algopy import *
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
#        dx/dt = f(t,x,p) = p1 + q0*x
#        x(0)  = x_0 = p0
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
#        q_* = argmin_{mu \in MU} E_mu[ Phi(Q)]
#        where mu is a probability measure
#        MU a familily of probability measures
#        Q a random variable, i.e. a functional Q(\omega): L^2(-\infty,\infty] ---> R
#        HERE:    MU family of all uniform distributions U(q - sigma , q + sigma)
#
#
"""

# DEFINING THE OED PROBLEM AND IMPLEMENTATION OF THE EXPLICIT EULER ODE INTEGRATOR
# to solve the problem, an ode integrator is needed, we implement here 
# the explicit euler method, since it is the simplest, it can also be differentiated
# with adol-c	
	
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
	tmp1 = ((J.T).dot(J))
	return (tmp1.inv()).trace()

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
	DM = 4       # degree of moments

	# generate pseudo measurement data
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
	Jtmp = adolc.jacobian(2,v)[:,:2]
	Jshp = shape(Jtmp)
	J = zeros((DM-1,1,Jshp[0],Jshp[1]))
	J[0,0,:,:] = Jtmp
	cg = CGraph()

	FJ = Function(Mtc(J))
	Ff = Phi(FJ)
	cg.independentFunctionList = [FJ]
	cg.dependentFunctionList = [Ff]
	#cg.plot('testgraph.png')
	#cg.plot('testgraph.svg')

	#def gradient_of_PHI(v):
		#""" computes grad(PHI) as needed in the steepest descent optimization"""
		#DM = 2
		#for np in range(Np):
			#D = DM-1
			#V = zeros((Nv,D))
			#V[np,0] = 1.
			#V[2,0]  = 0.
			#tmp = adolc.hos_forward(1,v,V,0)[1][:,:]
			#J[0,0,:,np] = tmp[:,0]
			#J[1,0,:,np] = tmp[:,1]
			#J[2,0,:,np] = tmp[:,2]

		#Jtc=Mtc(J)

		## 2: forward evaluation of Phi
		#cg.forward([Jtc])
	
		## 3: reverse evaluation of Phi
		#Phibar = zeros((DM,1,1,1))
		#Phibar[0,0,0,0]=1.
		#cg.reverse([Mtc(Phibar)])

		#Jbar = FJ.xbar.TC[:,0,:,:]

		## 4: reverse evaluation of J
		#vbar = zeros(Nv)
		#for np in range(Np):
			#D = DM-1
			#keep = D+1
			#V = zeros((Nv,D))
			#V[np,0] = 1.
			#V[2,0]  = 0.
			#adolc.hos_forward(1,v,V,keep)
			#U = zeros((1,Nm,D))
			## U is a (Q,M,D) array
			## Jbar is a (D,M,Np) array
			#U[0,:,0] = Jbar[0,:,np]
			#U[0,:,1] = Jbar[1,:,np]
			#U[0,:,2] = Jbar[2,:,np]
			#print 'U=',U
			#Z = adolc.hov_ti_reverse(1,U)[0]
			##print 'Z=',Z
	
			#vbar += Z[0,2,1]
		#return vbar

	def gradient_of_E_PHI(v,DM):
		""" computes the gradient of the expectation of PHI, i.e. grad( E[PHI] ),
			where E[PHI] is approximated by the method of moments up to moment degree DM.
		    This gradient is then used in the steepest descent optimization below"""

		# multi indices for the full tensor of order DM-1 to compute d^(DM-1) PHI
		# the remaining order to compute d^DM PHI is achieved by a reverse sweep later
		Jq = generate_multi_indices(Nq,DM)
		NJq = shape(Jq)[0]

		# since Nq is usually much larger than Np
		# we want to be able to get one higher order derivative
		# w.r.t. q by reverse mode
		# that means we have to evaluate the mixed partial derivatives
		# d^2 F/dpdq in the forward mode
		# since UTPS cannot do those partial derivatives straightforward
		# we need to use interpolation

		J = zeros((DM+1,1,Nm,Np)) # Taylor series of degree DM has DM+1 taylor coefficients (i.e. count also the zero-derivative)

		for nq in range(NJq):
			# 1: evaluation of J

			# here, we have to use nested interpolation
			# we need to compute
			# d^d F(x + s_1 z_1 + s_2 z_2) |
			# ---------------------------- |
			#   d z_1 d^(d-1) z_2          |_z=0
			# for s_1 = [1,0,0] resp. s_1 = [0,1,0] (one direction for each parameter p)
			# we use here formula (13) of the paper "Evaluating higher derivative tensors by forward propagation of univariate Taylor series"


			for np in range(Np): # each column of J
				for dm in range(DM+1):
					I = array([1,dm])
					K = zeros((dm+1,2),dtype=int)
					K[:,0] = 1
					K[:,1] = range(dm+1)
					V = zeros((Nv,dm+1))
					for k in K:
						s1 = zeros(Nv)
						s1[np] = k[0]
						s2 = zeros(Nv)
						s2[Np:] =  k[1]*Jq[nq,:]
						V[:,0] =  s1+s2
						tmp = adolc.hos_forward(1,v,V,0)[1]
						J[dm,0,:,np] += (-1)**multi_index_abs( I - k) * multi_index_binomial(I,k) * tmp[:,dm]
						
			scale_factor = array([1./prod(range(1,d+1)) for d in range(DM+1)])
			for dm in range(DM+1):
				J[dm,:,:,:] *= scale_factor[dm]

			# 2: forward evaluation of Phi
			Jtc=Mtc(J[:,:,:,:])
			cg.forward([Jtc])
			# 3: reverse evaluation of Phi
			Phibar = zeros((DM+1,1,1,1))
			Phibar[0,0,0,0]=1.
			cg.reverse([Mtc(Phibar)])
			Jbar = FJ.xbar.TC[:,0,:,:]
			#print Jbar
			#print shape(Jbar)

			## 4: reverse evaluation of J
			vbar = zeros(Nv)

			for np in range(Np): # each column of J
				dm = DM
				I = array([1,dm])
				K = zeros((dm+1,2),dtype=int)
				K[:,0] = 1
				K[:,1] = range(dm+1)

				V = zeros((Nv,dm+1))
				for k in K:
					s1 = zeros(Nv)
					s1[np] = k[0]
					s2 = zeros(Nv)
					s2[Np:] =  k[1]*Jq[nq,:]
					V[:,0] =  s1+s2

					keep = dm + 2
					adolc.hos_forward(1,v,V,keep)[1]

					# U is a (Q,M,D) array
					# Jbar is a (D,M,Np) array
					U = zeros((1,Nm,keep))
					for d in range(dm+1):
						U[0,:,d] = Jbar[d,:,np]
					#print 'U=',U

					Z = adolc.hov_ti_reverse(1,U)[0]
					#print 'Z=',Z
					#J[dm,0,:,np] += (-1)**multi_index_abs( I - k) * multi_index_binomial(I,k) * tmp[:,dm]
					#print shape(Z[0,:,:])
					#exit()
					#print Z[0,:,dm+1]
					tmp =  1./prod(range(1,dm+2)) * (-1)**multi_index_abs( I - k) * multi_index_binomial(I,k) * Z[0,Np,DM+1]
					vbar += tmp
			#vbar += Z[0,2,1]
			return vbar
		

	#print 'gradient_of_PHI'
	#print gradient_of_PHI(v)
	#print 'gradient_of_E_PHI'
	#print gradient_of_E_PHI(v,0)
	
	# perform steepest descent optimization
	vbar = inf
	count = 0
	while numpy.linalg.norm(vbar)>10**-8:
		count +=1
		vbar = gradient_of_E_PHI(v,0)
		v[2:] -= vbar[2:]
	
	print 'number of iterations =',count
	print 'v_opt =',v
	print 'v0=',v0

	## plot Phi
	## --------
	#def dFdp(p,q,ts,Sigma, etas):
		#v = concatenate((p,q))
		#return adolc.jacobian(2,v)[:,:Np]
	
	#qs = linspace(-1,2,100)
	#Phis = []
	#for q in qs:
		#q = array([q])
		#J = dFdp(p,q,ts,Sigma, etas)
		#Phis.append(Phi(J))

	#figure()
	#qplot = plot(qs,Phis,'k')
	#optimal_plot = plot([v[2]], cg.dependentFunctionList[0].x.X[0] , 'go')
	#xlabel(r'$q$')
	#ylabel(r'$\Phi(q)$')
	#legend((qplot, optimal_plot), (r'$\Phi(q)$','computed optimal solution'))
	#title('Optimal Design of Experiments')
	#savefig('odoe_objective_function.png')
	#savefig('odoe_objective_function.eps')

	## plot state for initial value and optimal value
	## ----------------------------------
	#figure()
	#p0 = v0[:2]
	#q0 = array([v0[2]])
	#x0 = explicit_euler(p0[0],f,ts,p0,q0)
	#initial_plot = plot(ts,x0,'b')

	#p_opt = v[:2]
	#q_opt = array([v[2]])
	#x_opt = explicit_euler(p_opt[0],f,ts,p_opt,q_opt)
	#opt_plot = semilogy(ts,x_opt,'r')
	#xlabel(r'time $t$ [sec]')
	#ylabel(r'$x(t)$')
	#legend((opt_plot,initial_plot),('optimal state traj.', 'initial state traj.'))
	
	##show()
	
	








