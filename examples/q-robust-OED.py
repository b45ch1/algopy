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
#        q_* = argmin_{mu \in MU} E_mu[ Phi(Q)]
#        where mu is a probability measure
#        MU a familily of probability measures
#        Q a random variable, i.e. a functional Q(\omega): L^2(-\infty,\infty] ---> R
#        HERE:    MU family of all uniform distributions U(q - sigma , q + sigma)
#
#
"""


# HIGHER ORDER DERIVATIVE TENSORS BY INTERPOLATION
# the theory is explained on page 315 of the book "Evaluating Derivatives" by Andreas Griewank,
# Chapter 13, Subsection: Multivariate Tensors via Univariate Tensors

def generate_multi_indices(N,D):
	"""
	generates 2D array of all possible multi-indices with |i| = D
	e.g.
	N=3, D=2
	array([[2, 0, 0],
       [1, 1, 0],
       [1, 0, 1],
       [0, 2, 0],
       [0, 1, 1],
       [0, 0, 2]])
	i.e. each row is one multi-index.
	"""
	T = []
	def rec(r,n,N,D):
		j = r.copy()
		if n == N-1:
			j[N-1] = D - numpy.sum(j[:])
			T.append(j.copy())
			return
		for a in range( D - numpy.sum( j [:] ), -1,-1 ):
			j[n]=a
			rec(j,n+1,N,D)
	r = numpy.zeros(N,dtype=int)
	rec(r,0,N,D)
	return numpy.array(T)


def convert_multi_indices_to_pos(in_I):
	"""
	a multi-index [2,1,0] tells us that we differentiate twice w.r.t x[0] and once w.r.t
	x[1] and never w.r.t x[2]
	This multi-index represents therefore the [0,0,1] element in the derivative tensor.
	"""
	I = in_I.copy()
	M,N = numpy.shape(I)
	D = numpy.sum(I[0,:])
	retval = numpy.zeros((M,D),dtype=int)
	for m in range(M):
		i = 0
		for n in range(N):
			while I[m,n]>0:
				retval[m,i]=n
				I[m,n]-=1
				i+=1
	return retval

def gamma(i,j):
	""" Compute gamma(i,j), where gamma(i,j) is define as in Griewanks book in Eqn (13.13)"""
	N = len(i)
	retval = [0.]
		
	def binomial(z,k):
		""" computes z!/[(z-k)! k!] """
		u = int(numpy.prod([z-i for i in range(k) ]))
		d = numpy.prod([i for i in range(1,k+1)])
		return u/d
	
	def alpha(i,j,k):
		""" computes one element of the sum in the evaluation of gamma,
		i.e. the equation below 13.13 in Griewanks Book"""
		term1 = (1-2*(numpy.sum(abs(i-k))%2))
		term2 = 1
		for n in range(N):
			term2 *= binomial(i[n],k[n])
		term3 = 1
		for n in range(N):
			term3 *= binomial(D*k[n]/ numpy.sum(abs(k)), j[n] )
		term4 = (numpy.sum(abs(k))/D)**(numpy.sum(abs(i)))
		return term1*term2*term3*term4
		
	def sum_recursion(in_k, n):
		""" computes gamma(i,j).
			The summation 0<k<i, where k and i multi-indices makes it necessary to do this 
			recursively.
		"""
		k = in_k.copy()
		if n==N:
			retval[0] += alpha(i,j,k)
			return
		for a in range(i[n]+1):
			k[n]=a
			sum_recursion(k,n+1)
			
	# putting everyting together here
	k = numpy.zeros(N,dtype=int)
	sum_recursion(k,0)
	return retval[0]
		
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
	while numpy.linalg.norm(vbar)>10**-8:
		# 1: evaluation of J
		Jtc=Mtc(adolc.jacobian(1,v)[:,:Np],J1)

		# 2: forward evaluation of Phi
		cg.forward([Jtc])
		#print 'Phi=',cg.dependentFunctionList[0].x.X
	
		# 3: reverse evaluation of Phi
		cg.reverse([Mtc([[1.]],[[0.]])])
		Jbar = FJ.xbar.X

		# 4: reverse evaluation of J
		x = v
		D = 2
		keep = D+1
		V = zeros((Nv,D))
		vbar = zeros(Nv)
		for np in range(Np):
			V[np,0] = 1
			u = (Jbar.T)[np,:].copy()
			adolc.hos_forward(1,D,x,V,keep)
			Z = adolc.hos_reverse(1,D,u)
			V[np,0] = 0
			#print 'Z=',Z
			vbar += Z[2,1]
		#update v:  x_k+1 = v_k - g
		v[2:] -= vbar[2:]
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
	
	








