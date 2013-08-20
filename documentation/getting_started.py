"""

What is ALGOPY:
---------------

The purpose of ALGOPY is the efficient evaluation of higher order derivatives
in the forward and reverse mode of Algorithmic Differentiation (AD). Particular
focus are matrix valued functions as they often appear in statistically motivated
functions. E.g. the covariance matrix of a least squares problem requires the
computation::

    C = inv(dot(J.T,J))

where J(x) is a partial derivative of a function F.

The central idea of ALGOPY is the computation on Taylor polynomials with scalar
coefficientsand with matrix coefficients. These algorithms are primarily used for
Algorithmic Differentiation (AD)in the forward and reverse mode.

The focus are univariate Taylor polynomials over matrices (UTPM),implemented in
the class `algopy.utpm.UTPM`.

To allow the use of the reverse mode of AD a simple code tracer has been implemented in
`algopy.tracer`. The idea is to record the computation procedure in a datastructure s.t.
the control flow sequence can walked in reverse order.

ALGOPY is a research prototype where, to the best of authors'
knowledge, some algorithms are implemented that cannot be found elsewhere.

Most of ALGOPY is implemented in pure Python. However, some submodules are implemented
in pure C. For these submodules Python bindings using ctypes are provided.

The algorithms are quite well-tested and have been successfully used.
However, since the user-base is currently quite small, it is possible that bugs
may still be persistent.
Also, not all submodules of ALGOPY are feature complete. The most important parts
`algopy.tracer` and `algopy.upt.UTPM` are however fairly complete.



Getting Started:
----------------
Consider the following function:
"""

def f(A,x):
    for n in range(3):
        y = dot(x.T,dot(A,x))
        A = A - dot(x,x.T) * y
        
    return trace(A)
    
"""
and it is the goal to compute its gradient w.r.t. A and x.
As one will surely notice, this is not as simple as it seems.
But it's no problem for ALGOPY.

At first, we will find the gradient in the forward mode of AD.
Let A be an (N,N) array, and x an (N,1) array. Therefore, the gradient of f
will be a vector of length `N**2 + N`. In the forward mode one computes
each of those `N**2 + N` by a separate run. This is very similar to the finite differences
approach where each argument is separately perturbed.

As an example, let's compute df/dA_{11}, i.e. the derivative w.r.t. the (1,1) entry of A,
(counting from 1).
"""
import numpy
from algopy import UTPM
from algopy.globalfuncs import dot, trace
D,P,N = 2,1,2
A = UTPM(numpy.zeros((D,P,N,N)))
x = UTPM(numpy.zeros((D,P,N,N)))

A.data[0,:] = numpy.random.rand(N,N)
x.data[0,:] = numpy.random.rand(N,1)

A.data[1,0,0,0] = 1.
y = f(A,x)

print('df/dA_{11] = ',y.data[1,0])

"""
Of course it is the goal to obtain all partial derivatives. 
That's why there is the `P` argument above. One can simultaneously propagate
several directional derivatives at once. In our example we have N**2 elements in
A and N in x resulting in a total of P = N**2 + N directions. The 0'th coefficient
of all P directions have to be the same.
"""

D,N = 2,2
P = N**2 + N
A = UTPM(numpy.zeros((D,P,N,N)))
x = UTPM(numpy.zeros((D,P,N,1)))

A.data[0,:] = numpy.random.rand(N,N)
x.data[0,:,:,0] = numpy.random.rand(N)

for n1 in range(N):
    for n2 in range(N):
        A.data[1,n1*N + n2,n1,n2] = 1.

for n in range(N):
    x.data[1,N**2 + n,n,0] = 1.
  
# print A.data  
# print x.data
    
y = f(A,x)

gradient_AD = y.data[1,:]

print('AD gradient g(A,x) = ', gradient_AD)

"""
Ok, no idea if that's correct. However, we can check by Finite Differences if
the solution makes sense. We are going to use only the 0'th coefficient.
"""

A = A.data[0,0]
x = x.data[0,0]

epsilon = numpy.sqrt(2**-53) # mantissa in 64bit IEEE 754 floating point arithmetic is 53 bits

# rule of thumb how to pick the perturbation
deltaA = numpy.abs(A) * epsilon
deltax = numpy.abs(x) * epsilon

gradient_FD = [] # list with the finite differences solutions
for n1 in range(N):
    for n2 in range(N):
        tmp = A.copy()
        tmp[n1,n2] += deltaA[n1,n2]
        gradient_FD.append((f(tmp,x) - f(A,x))/ deltaA[n1,n2] )

for n in range(N):
    tmp = x.copy()
    tmp[n] += deltax[n]
    gradient_FD.append((f(A,tmp) - f(A,x))/ deltax[n] )
    

gradient_FD = numpy.ravel(gradient_FD)

"""
Checking now how close the AD solution is to FD. The FD solution is at best
numpy.sqrt(2**-53) digits accurate.
"""

print('gradient_FD - gradient_AD = ', gradient_FD - gradient_AD) 



