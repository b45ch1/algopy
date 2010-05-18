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
D,N = 1,3
P = 5

x = UTPM(numpy.random.rand(*(1,P,2*N)))

# wrap the UTPM instance in a Function instance to trace all operations 
# that have x as an argument
# x = Function(x)

v1 = x # x[N:]
v2 = x # x[:N]
# v1 = x[N:]
# v2 = x[:N]
v3 = dot(v1,v2)
# print 'x.data=',x.data

# print 'v1.data=',v1.data
# print 'v2.data=',v2.data
# print 'v3.data=',v3.data

print 'v1.data.strides=',v1.data.strides
print 'v2.data.strides=',v2.data.strides
print 'v3.data.strides=',v3.data.strides

print 'v1.data.shape=',v1.data.shape
print 'v2.data.shape=',v2.data.shape
print 'v3.data.shape=',v3.data.shape

y = v3 * v1


# y = f(x,N)

# print y.shape

# cg.independentFunctionList = [x]
# cg.dependentFunctionList = [y]

# print cg


