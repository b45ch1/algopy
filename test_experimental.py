from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utpm import *
from algopy.globalfuncs import *

# first order derivatives, one directional derivative
# D - 1 is the degree of the Taylor polynomial
# P directional derivatives at once
# M number of rows of J1
# N number of cols of J1
# K number of rows of J2 (must be smaller than N)
D,P,M,N,K,Nx = 2,1,100,3,1,1


# METHOD 1: nullspace method
cg1 = CGraph()

J1 = Function(UTPM(numpy.random.rand(*(D,P,M,N))))
J2 = Function(UTPM(numpy.random.rand(*(D,P,K,N))))


Q,R = Function.qr_full(J2.T)
Q2 = Q[:,K:].T

J1_tilde = dot(J1,Q2.T)
Q,R = qr(J1_tilde)
V = solve(R.T, Q2)
C = dot(V.T,V)
cg1.trace_off()

cg1.independentFunctionList = [J1, J2]
cg1.dependentFunctionList = [C]

print 'covariance matrix: C =\n',C
print 'check that Q2.T spans the nullspace of J2:\n', dot(J2,Q2.T)

# METHOD 2: image space method (potentially numerically unstable)
cg2 = CGraph()

J1 = Function(J1.x)
J2 = Function(J2.x)

M = Function(UTPM(numpy.zeros((D,P,N+K,N+K))))
M[:N,:N] = dot(J1.T,J1)
M[:N,N:] = J2.T
M[N:,:N] = J2
C2 = inv(M)[:N,:N]
cg2.trace_off()

cg2.independentFunctionList = [J1, J2]
cg2.dependentFunctionList = [C2]

print 'covariance matrix: C =\n',C2
print 'difference between image and nullspace method:\n',C - C2

Cbar = UTPM(numpy.random.rand(D,P,N,N))

cg1.pullback([Cbar])

cg2.pullback([Cbar])
print 'derivatives w.r.t. J1 correct? \n',cg2.independentFunctionList[0].xbar - cg1.independentFunctionList[0].xbar
print 'derivatives w.r.t. J2 correct? \n',cg2.independentFunctionList[1].xbar - cg1.independentFunctionList[1].xbar












