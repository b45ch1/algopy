#!/usr/bin/env python
from pylab import *
import sys
sys.path = ['..'] + sys.path
from matrix_ad import *
import adolc
import numpy.random
import scipy.optimize

"""
goal: compute the Hessian of the function

Phi(F) = trace(F*F)
F = [[x*y,x**2],[x**2*y,y**3*x]]
"""


# OBJECTIVE FUNCTION
# ------------------
def Phi(F):
	return trace(F*F)

def ffcn(x):
	 return 0.5*array(
[[x[0]*x[0], x[0]*x[0]],
[ x[0]*x[0] , x[1]*x[1]]])

# TAPING THE FUNCTIONS
# --------------------
# taping function ffcn
u = 3.; v = 7.
ax = array([adolc.adouble(u), adolc.adouble(v)])
adolc.trace_on(1)
ax[0].is_independent(u)
ax[1].is_independent(v)
ay = ffcn(ax)
for n in range(2):
	for m in range(2):
		adolc.depends_on(ay[n,m])
adolc.trace_off()

# taping matrix functions with algopy
x = array([u,v])
F = ffcn(x)
Fdot = zeros((2,2))
cg = CGraph()
FF = Function(Mtc(F))
Fy = Phi(FF)
cg.independentFunctionList = [FF]
cg.dependentFunctionList = [Fy]

# COMPUTING THE HESSIAN H = d^2 Phi/ dx^2
# ---------------------------------------
# need for that to propagate two directions
# then reverse

H = zeros((2,2))

for n in range(2):
	# 1: hos_forward, propagate two directions
	x = array([5.,2.])
	D = 2
	keep = D+1
	V = zeros((2,1))
	F = zeros((2,2))
	Fdot = zeros((2,2))
	V[n,0] = 1.
	(y,W) = adolc.hos_forward(1,D,x,V,keep)
	V[n,0] = 0.
	F[0,:] = y[:2]
	F[1,:] = y[2:]
	#print 'W=',W
	Fdot[0,:] = W[:2,0]
	Fdot[1,:] = W[2:,0]

	#print 'F=',F
	#print 'Fdot=',Fdot
	
	# 2: matrix forward
	cg.forward([Mtc(F,Fdot)])
	#print 'cg.dependentFunctionList[0].x',cg.dependentFunctionList[0].x
	
	# 3: matrix reverse
	Phibar = array([[1.]])
	Phibardot = array([[0.]])
	cg.reverse([Mtc(Phibar, Phibardot)])
	#print '----------------'
	#print cg
	#print 'cg.independentFunctionList[0].xbar',cg.independentFunctionList[0].xbar

	
	# 4: hov_reverse
	U = zeros((2,4))
	U[0,:] = cg.independentFunctionList[0].xbar.X.flatten()
	U[1,:] = cg.independentFunctionList[0].xbar.Xdot.flatten()
	#print 'U=',U
	#print 'adolc.hov_reverse(1,D,U)[0]=', adolc.hov_reverse(1,D,U)[0][:,:,:]
	res = adolc.hov_reverse(1,D,U)[0].copy()
	#print res[0,:,:]
	#print res[1,:,:]
	res[0,:,1:] += res[1,:,:-1]
	#print res[0,:,:]
	H[n,:] = res[0,:,1]

	#tmp1 = adolc.hov_reverse(1,D,U)[0][:,:,:]

print H
#for n in range(2):
	#for m in range(2):
		#Fdot[n,m] = 1
		#cg.forward([Mtc(F,Fdot)])
		#Phibar = array([[1.]])
		#Phibardot = array([[0.]])
		#cg.reverse([Mtc(Phibar, Phibardot)])
		#Fdot[n,m] = 0
		##print cg.independentFunctionList[0].xbar

# compute new search direction
#delta_q = numpy.linalg.solve(H,-g[:,0])
#print delta_q
#q_plus = [13.,17.] + delta_q
#assert numpy.prod(q_plus == [0.,0.])