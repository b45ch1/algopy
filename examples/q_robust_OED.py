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
	Nm = size(ts)
	Nx = size(x0)
	if isinstance(p[0],adolc.adouble):
		x = array([[adolc.adouble(0) for n in range(Nx)]  for m in range(Nm)])
	else:
		x = zeros((Nm,Nx))
	x[0,:] = x0[:]
	for m in range(1,Nm):
		h = ts[m] - ts[m-1]
		x[m,:]= x[m-1,:] + h*f(ts[m-1],x[m-1,:],p,q)
	return x

def f(t,x,p,q):
	""" rhs of the ODE"""
	return array([p[1] + q[0]*x[0], q[0]*x[1], 1. + q[0]*x[2]])

def f2(t,x,p,q):
	""" rhs of the ODE"""
	return array([p[1] + p[0]*q[0]*x[0], q[0]*x[0] + p[0]*q[0]*x[1], 1. + p[0]*q[0]*x[2]])

def measurement_model(x,p,q):
	return x

def F(p,q,ts,Sigma, etas):
	x0 = array([v[0], 1., 0.])
	x = explicit_euler(x0,f2,ts,p,q)
	h = measurement_model(x[:,0],p,q)
	return dot(Sigma, h-etas)

def dFdp(p,q,ts,Sigma, etas):
	x0 = array([v[0], 1., 0.])
	x = explicit_euler(x0,f2,ts,p,q)
	h = measurement_model(x[:,1:],p,q)
	return dot(Sigma,h)

def Phi(J):
	""" prototypical OED objective function"""
	tmp1 = ((J.T).dot(J))
	return (tmp1.inv()).trace()

def unifom_distr_moment(d,sigma):
	""" computes the moments E[X] for X uniformly (-sigma,sigma) distributed """
	if 1==d%2:
		return 0
	else:
		return sigma**d/(d+1.)

if __name__ == "__main__":
	# problem setup
	Nm = 100    # number of measurements
	Np = 2      # number of parameters
	Nq = 1      # number of control variables
	Nv = Np + Nq
	ts = linspace(0,3,Nm)
	Sigma = eye(Nm)
	p = array([1.,2.])
	q = array([-1.])
	v = concatenate((p,q))
	DM = 4       # degree of moments
	sigma = 0.1  # "deviation" of the uniform distribution

	# test_explicit_euler_integration
	x0 = array([v[0], 1., 0.])
	x = explicit_euler(x0, f2, ts, v[:Np], v[Np:])
	figure()
	plot(ts,x[:,0])
	plot(ts,x[:,1])
	plot(ts,x[:,2])
	
	# generate pseudo measurement data
	p[0]+=3.; 	p[1] += 2.
	x0 = array([v[0], 1., 0.])
	x = explicit_euler(x0,f2,ts,p,q)
	h = measurement_model(x[:,0],p,q)
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

	# taping dFdp
	av = array([adolc.adouble(0) for i in range(Nv)])
	adolc.trace_on(1)
	av[0].is_independent(p[0])
	av[1].is_independent(p[1])
	av[2].is_independent(q[0])
	ay = dFdp(av[:Np],av[Np:],ts,Sigma,etas)
	for m in range(Nm):
		for n in range(Np):
			adolc.dependent(ay[m,n])
	adolc.trace_off()

	# tape the objective function with Algopy
	J = zeros((DM-1,1, Nm, Np))
	J[0,0,:,:] = numpy.random.rand(Nm,Np)
	cg = CGraph()
	FJ = Function(Mtc(J))
	Ff = Phi(FJ)
	cg.independentFunctionList = [FJ]
	cg.dependentFunctionList = [Ff]	
	

	# PERFORM OED
	# ===========
	v0 = v.copy()

	def expectation_of_phi(v,DM,sigma):
		
		# STEP1:  forward UTPS through dFdp
		V = zeros((Nv,max(DM,1)))
		V[Np,0] = 1.
		(y,W) = adolc.hos_forward(1,v,V,0)
		y = y.reshape((Nm,Np))
		W = W.reshape((Nm,Np,shape(W)[1]))
		J = zeros((DM+1,1,Nm,Np))

		# fill 0th degree of J
		J[0,0,:,:] = y

		# fill d'th degree of J
		for dm in range(1,DM+1):
			J[dm, 0,:,: ] = W[:,:,dm-1]


		# STEP2:  reverse UTPM through PHI
		Jtc=Mtc(J[:,:,:,:])
		cg.forward([Jtc])


		retval = 0.
		for dm in range(DM+1):
			retval += unifom_distr_moment(dm,sigma) * Ff.x.TC[dm,0,0,0]
		return retval
		


	# plot objective function
	figure()

	for dm in range(0,7,2):
		print dm
		Nqs = 200
		qs = linspace(-1,2,Nqs)
		Phis = zeros(Nqs)
		for n in range(Nqs):
			v = array([p[0],p[1],qs[n]])
			Phis[n] = expectation_of_phi(v,dm,sigma)
		semilogy(qs,Phis)
	show()
	





