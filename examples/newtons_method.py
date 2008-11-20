#!/usr/bin/env python
from pylab import *
import sys
sys.path = ['..'] + sys.path
from matrix_ad import *
import adolc
import numpy.random
import scipy.optimize
import numpy.linalg


"""
goal: compute the Hessian of the function

Phi(F) = trace(F*F)
F = [[x*y,x**2],[x**2*y,y**3*x]]
"""


# OBJECTIVE FUNCTION
# ------------------
def Phi(F):
	return trace( dot(F.T,F))

def ffcn(x):
	 return 0.5*array(
[[(x[0]-17.)*(x[0]-17.), (x[0]-17.)*(x[0]-17.)],
[ x[1]-19. , x[1]-19.]])

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

def gradient_and_hessian_of_Phi(x):
	H = zeros((2,2)) # Hessian
	g = zeros(2) # gradient
	V = zeros((2,1))
	F = zeros((2,2))
	Fdot = zeros((2,2))
	D = 2
	keep = D+1
	
	for n in range(2):
		# 1: hos_forward, propagate two directions
		V[n,0] = 1.
		(y,W) = adolc.hos_forward(1,D,x,V,keep)
		V[n,0] = 0.
		F[0,:] = y[:2]
		F[1,:] = y[2:]
		Fdot[0,:] = W[:2,0]
		Fdot[1,:] = W[2:,0]

		# 2: matrix forward
		cg.forward([Mtc(F,Fdot)])

		# 3: matrix reverse
		Phibar = array([[1.]])
		Phibardot = array([[0.]])
		cg.reverse([Mtc(Phibar, Phibardot)])

		# 4: hov_reverse
		U = zeros((1,4,2))
		U[0,:,0] = cg.independentFunctionList[0].xbar.X.flatten()
		U[0,:,1] = cg.independentFunctionList[0].xbar.Xdot.flatten()
		res = adolc.hovt_reverse(1,D,U)[0].copy()
		g[:]   = res[0,:,0]
		H[n,:] = res[0,:,1]
		
	return (g,H)

def newtons_method(x0):
	x = x0.copy()

	g = numpy.inf
	k = 0
	while numpy.linalg.norm(g)>10**-12:
		print 'iteration: %2d'%k; k+=1
		(g,H) = gradient_and_hessian_of_Phi(x)
		# compute new search direction
		delta_x = numpy.linalg.solve(H,-g)
		#update x
		x += delta_x
	return x

x = numpy.array([13.,17.])
print newtons_method(x)
