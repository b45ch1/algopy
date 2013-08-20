"""
This example shows how ALGOPY can be used for linear error propagation.

Consider the error model::

    y = x + \epsilon

where x a vector and \epsilon a random variable with zero mean and
covariance matrix \Sigma^2. The y is the observed quantity and x is a real vector
representing the "true" value.

One can find some estimator \hat x that is in some or another way optimal.
For instance one take 100 samples and obtain y_1,y_2,....,y_100 and take the
arithmetic mean as an estimator for x. In the following we simply assume that 
some estimate \hat x is known and has an associated confidence region described 
by its covariance matrix Sigma^2 = E[(\hat x - E[\hat x])(\hat x - E[\hat x])^T]

However, not \hat x is of interest but some function f(\hat x)::

    f: R^N ---> R^M
        \hat x ---> \hat x = f(\hat x)

The question is:

    What can we say about the confidence region of the function f(y) when
    the confidence region of y is described by the covariance matrix \Sigma^2?

For affine (linear) functions::

    z = f(y) = Ay + b
        
the procedure is described in the 
wikipedia article http://en.wikipedia.org/wiki/Propagation_of_uncertainty .

For nonlinear functions can be linearized about an estimate \hat y of E[y].
In the vicinity of \hat y, the linear model approximates the nonlinear function often quite well.

To linearize the function, the Jacobian J(\hat y) of the function f(\hat y) has to be computed, i.e.:

    z \approx f(y) = f(\hat y) + J(\hat y) (y - \hat y)

The covariance matrix of z is defined as C = E[z z^T] = E[ J y y^T J^T] = J \Sigma^2 J^T.
That means if we know J(y), we can approximately compute the confidence region if
f(\hat y) is sufficiently linear.

To compute the Jacobian one can use the forward and the reverse mode of AD:
In the forward mode of AD one computes Nm directional derivatives, i.e. P = Nm.
In the reverse mode of AD one computes M adjoint derivatives, i.e. Q = M.
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
print('Jacobian J(y) = \n', J)

C_epsilon = 0.3*numpy.eye(Nm)

print(J.shape)

C = dot(J.T, dot(C_epsilon,J))

print('Covariance matrix of z: C = \n',C)
        
    
        

    
 
