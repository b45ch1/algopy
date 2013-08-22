#!/usr/bin/env python
import sys
sys.path.append('../')
from pylab import *
from prettyplotting import *
from adolc import *
import adolc
from algopy import *
from numpy import *
from time import *

def myqr(in_A):
	# input checks
	Ndim = ndim(in_A)
	assert Ndim == 2
	N,M = shape(in_A)
	assert N==M

	# algorithm
	R = in_A.copy()
	if type(R[0,0]) == adouble:
		QT = array([[adouble(n==m) for m in range(N)] for n in range(N) ])
	else:
		QT = eye(N)

	for n in range(N):
		for m in range(n+1,N):
			a = R[n,n]
			b = R[m,n]
			r = sqrt(a**2 + b**2)
			c = a/r
			s = b/r

			for k in range(N):
				Rnk = R[n,k]
	
				R[n,k] = c*Rnk + s*R[m,k]
				R[m,k] =-s*Rnk + c*R[m,k];

				QTnk = QT[n,k]
				QT[n,k] = c*QTnk + s*QT[m,k]
				QT[m,k] =-s*QTnk + c*QT[m,k];
			#print 'QT:\n',QT
			#print 'R:\n',R
			#print '-------------'

	return QT,R
			
def inv(in_A):
	QT,R = myqr(in_A)
	N = shape(in_A)[0]

	for n in range(N-1,-1,-1):
		Rnn = R[n,n]
		R[n,:] /= Rnn
		QT[n,:] /= Rnn
		for m in range(n+1,N):
			Rnm = R[n,m]
			R[n,m] = 0
			QT[n,:] -= QT[m,:]*Rnm

	return QT,R

def print_loc(aA):
	print('---')
	for n in range(N):
		print('[', end=' ')
		for m in range(N):
			print(aA[n,m].loc," ", end=' ')
		print(']')

if __name__ == "__main__":
	from numpy.random import random

	Ns = list(range(1,22))
	adolc_times = []
	adolc_taping_times = []
	adolc_num_operations = []
	adolc_num_locations = []
	
	algopy_times = []
	

	for N in Ns:
		print('N=',N)
		A = random((N,N))
		#A = array([
		#[0.018 ,0.0085 ,0.017 ,0.017],
		#[0.02 ,0.0042 ,0.0072 ,0.016],
		#[0.006 ,0.012 ,0.01 ,0.014],
		#[0.0078 ,0.011 ,0.02 ,0.02]], dtype= float64)

		# with ADOL-C
		# -----------
		N = shape(A)[0]

		t_start = time()
		aA = array([[adouble(A[n,m]) for m in range(N)] for n in range(N) ])

		trace_on(0)
		for n in range(N):
			for m in range(N):
				independent(aA[n,m])
		aC = inv(aA)[0]
		ay = trace(aC)
		dependent(ay)
		trace_off()

		adolc_num_locations.append(tapestats(0)['NUM_LOCATIONS'])
		adolc_num_operations.append(tapestats(0)['NUM_OPERATIONS'])
	
		t_end = time()
		adolc_taping_times.append(t_end-t_start)
		t_start = time()
		H1 = adolc.hessian(0,A.ravel())
		t_end = time()
		print('adolc needs %0.6f seconds'%(t_end - t_start))
		adolc_times.append(t_end-t_start)

		# with ALGOPY

		t_start = time()
		cg = CGraph()
		
		FA = zeros((2,N**2,N,N))
		for n in range(N):
			for m in range(N):
				FA[0,n*N+m,:,:] = A[:,:]
				FA[1,n*N+m,n,m] = 1.

		
		FA = Mtc(FA)
		FA = Function(FA)
		FC = FA.inv()
		Fy = FC.trace()
		cg.independentFunctionList = [ FA ]
		cg.dependentFunctionList = [ Fy ]

		ybar = zeros((2,N**2,1,1))
		ybar[0,:,0,0] = 1.

		cg.reverse([Mtc(ybar)])

		# put everything in a Hessian
		H2 = zeros((N**2,N**2))
		for n in range(N**2):
			H2[n,:] = FA.xbar.TC[1,n,:].ravel()
		
		t_end = time()
		print('algopy needs %0.6f seconds'%(t_end - t_start))
		algopy_times.append(t_end-t_start)

		#print H1 - H2

	figure()
	semilogy(Ns,adolc_taping_times,'ko', label='pyadolc: taping')
	semilogy(Ns,adolc_times,'k.', label='pyadolc: hessian computation')
	semilogy(Ns,algopy_times,'kd', label='algopy: hessian computation')
	legend(loc=4)
	title('UTPS vs UTPM for Hessian Computation of $y =$tr$X^{-1}$')
	xlabel('matrix size $N$')
	ylabel('runtime $t$ in seconds')
	
	savefig('utps_vs_utpm.png')
	savefig('utps_vs_utpm.eps')

	figure()
	semilogy(Ns,adolc_num_operations,'k.', label='Number of Operations')
	semilogy(Ns,adolc_num_locations,'ko', label='Number of Locations')

	title('Memory needed by PyADOLC')
	xlabel('matrix size $N$')
	ylabel('size')

	savefig('pyadolc_locs_and_ops.png')
	savefig('pyadolc_locs_and_ops.eps')
	legend(loc=4)

	
	show()