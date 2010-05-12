"""
This example shows how ALGOPY can be used for linear error propagation.

Consider the error model::

    y = x + \epsilon

where x,y,\epsilon arrays of size Nm. \epsilon is a iid normally distributed
random variable with mean zero and covariance matrix \Sigma^2.
The y are the observed quantities and x a real vector.

The question is:

    What can we say about the confidence region of a function f(y) when
    the confidence region of y is described by the covariance matrix \Sigma^2?

The idea is to compute the Jacobian J(\hat x) of a function::

    f: R^Nm ---> R^M
        y  ---> z = f(y)
        
Since y depends on the random variable \epsilon it is also a random variable 
with the same covariance matrix as \epsilon.

The covariance matrix of z, i.e. E[z z^T] = E[ J y y J^T] = J \Sigma^2 J^T.
I.e., if we know J(y) we can approximately compute the confidence region if
f(y) is sufficiently linear.

In the forward mode of AD we compute Nm directional derivatives, i.e. P = Nm.
"""

import numpy
from algopy import CGraph, Function, UTPM, dot, qr, eigh, inv, zeros

def f(y):
    retval = zeros((3,1),dtype=y)
    retval[0,0] = numpy.log(dot(y.T,y))
    retval[1,0] = numpy.exp(dot(y.T,y))
    retval[2,0] = numpy.exp(dot(y.T,y)) -  numpy.log(dot(y.T,y))
    return retval
    
D,Nm = 2,40
P = Nm
y = UTPM(numpy.zeros((2,P,Nm)))

y.data[0,:] = numpy.random.rand(Nm)
y.data[1,:] = numpy.eye(Nm)



# print f(y)

J = f(y).data[1,:,:,0]

print 'Jacobian J(y) = \n', J

C_epsilon = 0.3*numpy.eye(Nm)

print J.shape

C = dot(J.T, dot(C_epsilon,J))

print 'Covariance matrix of z: C = \n',C
        
    
        

    
 
