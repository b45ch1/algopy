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


# objective function
def Phi(F):
	return trace(F*F)

def ffcn(x):
	 return array(
[[x[0]*x[1], x[0]**2],
[ x[0]**2 * x[1], x[1]**3 * x[0]])



# TAPING FUNCTION F
u = 3.; v = 7.
ax = array([adouble(u), adouble(v)])
adolc.trace_on(1)
ax[0].is_independent(u)
ax[1].is_independent(v)
ay = ffcn(ax)
for n in range(2):
	for m in range(2):
		adolc.depends_on(ay[n,m])
adolc.trace_off()

# TAPING MATRIX FUNCTIONS WITH ALGOPY
cg = CGraph()
FF = Function(Mtc(F))
Fy = Phi(FF)
cg.independentFunctionList = [FF]
cg.dependentFunctionList = [Fy]

for n in range(2):
	for m in range(2):
		Fdot[n,m] = 1
		cg.forward([Mtc(F,Fdot)])
		Phibar = array([[1.]])
		Phibardot = array([[0.]])
		cg.reverse([Mtc(Phibar, Phibardot)])
		Fdot[n,m] = 0
		print cg.independentFunctionList[0].xbar

# compute new search direction
#delta_q = numpy.linalg.solve(H,-g[:,0])
#print delta_q
#q_plus = [13.,17.] + delta_q
#assert numpy.prod(q_plus == [0.,0.])