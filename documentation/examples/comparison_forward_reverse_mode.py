"""
We show here how the forward and the reverse mode of AD are used and show
that they produce the same result.

We consider the function f:R^N ---> R defined by

def f(x,y):
    return dot(x,y) - x*(x-y)
    
We want to compute the Hessian of that function.

"""

import numpy
from algopy import CGraph, Function, UTPM, dot, qr, eigh, inv, zeros


def f(x,N):
    return dot(x[:N],x[N:])*x[N:]  - x[:N]*(x[:N]-x[N:])
    
# create a CGraph instance that to store the computational trace
cg = CGraph()

# create an UTPM instance
D,N,M = 2,3,2
P = N

A = UTPM(numpy.zeros((D,P,M,N)))
x = UTPM(numpy.zeros((D,P,N,1)))

x.data[0,:] = numpy.random.rand(N,1)
A.data[0,:] = numpy.random.rand(M,N)

x.data[1,:,:,0] = numpy.eye(P)


x = Function(x)
A = Function(A)


# wrap the UTPM instance in a Function instance to trace all operations 
# that have x as an argument
# x = Function(x)

y = dot(A,x)

# define dependent and independent variables in the computational procedure
cg.independentFunctionList = [x,A]
cg.dependentFunctionList = [y]

# for such linear function we already know the Jacobian: df/dx = A
# y.data is a (D,P,N) array, i.e. we have to transpose to get the Jacobian
# Since the UTPM instrance is wrapped in a Function instance we have to access it
# by y.x. That means the Jacobian is
J = y.x.data[1].T

# # checking against the analytical result
print('J - A =\n', J - A.x.data[0,0])

# Now we want to compute the same Jacobian in the reverse mode of AD
# before we do that we have a look what the computational graph looks like:
# print 'Computational graph is', cg

# the reverse mode is called by cg.pullback([ybar])
# it is a little hard to explain what's going on here. Suffice to say that we
# now compute one row of the Jacobian instead of one column as in the forward mode

ybar = y.x.zeros_like()

# compute first row of J
ybar.data[0,0,0,0] = 1
cg.pullback([ybar])
J_row1 = x.xbar.data[0,0]

# compute second row of J
ybar.data[...] = 0
ybar.data[0,0,1,0] = 1
cg.pullback([ybar])
J_row2 = x.xbar.data[0,0]

# build Jacobian
J2 = numpy.vstack([J_row1.T, J_row2.T])
print('J - J2 =\n', J - J2)

# one can also easiliy extract the Hessian which is here a (M,N,N)-tensor
# e.g. the hessian of y[1] is zero since y[1] is linear in x
print('Hessian of y[1] w.r.t. x = \n',x.xbar.data[1,:,:,0])









