import numpy
import algopy
from algopy import CGraph, UTPM, Function

def eval_g(x, y):
    """ some vector-valued function """
    retval = algopy.zeros(3, dtype=x)
    retval[0] = algopy.sin(x**2 + y)
    retval[1] = algopy.cos(x+y) - x
    retval[2] = algopy.sin(x)**2 + algopy.cos(x)**2
    return retval

# trace the function evaluation
# and store the computational graph in cg
cg = CGraph()
ax = 3.
ay = 5.
fx = Function(ax)
fy = Function(ay)
fz = eval_g(fx, fy)
cg.independentFunctionList = [fx, fy]
cg.dependentFunctionList = [fz]

# compute Taylor series
#
#  Jx( 1. + 2.*t + 3.*t**2 + 4.*t**3 + 5.*t**5,
#      6. + 7.*t + 8.*t**2 + 9.*t**3 + 10.*t**5 )
#  Jy( 1. + 2.*t + 3.*t**2 + 4.*t**3 + 5.*t**5,
#      6. + 7.*t + 8.*t**2 + 9.*t**3 + 10.*t**5 )
#
# where
#
# Jx = dg/dx
# Jy = dg/dy


# setup input Taylor polynomials
D,P = 5, 3  # order D=5, number of directions P
ax = UTPM(numpy.zeros((D, P)))
ay = UTPM(numpy.zeros((D, P)))
ax.data[:, :] = numpy.array([1., 2. ,3., 4. ,5.]).reshape((5,1))  # input Taylor polynomial
ay.data[:, :] = numpy.array([6., 7. ,8., 9. ,10.]).reshape((5,1))  # input Taylor polynomial

# forward sweep
cg.pushforward([ax, ay])

azbar = UTPM(numpy.zeros((D, P, 3)))
azbar.data[0, ...] = numpy.eye(3)

# reverse sweep
cg.pullback([azbar])

# get results
Jx = cg.independentFunctionList[0].xbar
Jy = cg.independentFunctionList[1].xbar

print('Taylor series of Jx =\n', Jx)
print('Taylor series of Jy =\n', Jy)
