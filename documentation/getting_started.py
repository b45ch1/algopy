"""

What is ALGOPY:
---------------

The central idea of ALGOPY is the computation on Taylor polynomials with scalar coefficients
and with matrix coefficients. These algorithms are primarily used for Algorithmic Differentiation (AD)
in the forward and reverse mode. The focus are univariate Taylor polynomials over matrices (UTPM),
implemented int the class `algopy.utpm.UTPM`.

To allow the use of the reverse mode of AD a simple code tracer has been implemented in
`algopy.tracer`. The idea is to record the computation procedure in a datastructure s.t.
the control flow sequence can walked in reverse order.

ALGOPY is a research prototype where, to the best of authors'
knowledge, some novel algorithms are implemented. In particular, the algorithms for
the qr decomposition of matrix polynomials and the symmetric eigenvalue decomposition with
possibly repeated eigenvalues are novel.

Most of ALGOPY is implemented in pure Python. However, some submodules are implemented
in pure C. For these submodules Python bindings using ctypes are provided.


A word of warning:
------------------
There is a good chance that the  *user* API might change in the future to make
the software even more intuitive and easy to use.
We will try to stay backward-compatible if possible but we rate clean code over
backward-compatibility at the current stage.
The algorithms are quite well-tested and have been successfully used.
However, since the user-base is currently quite small, it is possible that bugs
may still be persistent.
Also, the current code is by far from feature complete. In particular, all 
algorithms in `algopy.utps.UTPS` should be also available in `algopy.utpm.UTPM`,
but they are not.



Getting Started:
----------------
Consider the following function:
"""

def f(A,x):
    for n in range(50):
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

A.data[0,0] = numpy.random.rand(N,N)
x.data[0,0] = numpy.random.rand(N,1)

A.data[1,0,0,0] = 1.
y = f(A,x)

print 'df/dA_{11] = ',y.data[1,0]

"""
Of course we want all partial derivatives. 
That's why there is the `P` argument above. One can simultaneously propagate
several directional derivatives at once.
"""

D,P,N = 2,5,2
A = UTPM(numpy.zeros((D,P,N,N)))
x = UTPM(numpy.zeros((D,P,N,N)))

A.data[0,:] = numpy.random.rand(P,N,N)
x.data[0,:] = numpy.random.rand(P,N,1)

for n1 in range(N):
    for n2 in range(N):
        A.data[1,n1*N + n2,n1,n2] = 1.

for n in range(N):
    x.data[1,n,n] = 1.
    
print x
    
# y = f(A,x)

# print 'gradient g(A,x) = ', y.data[1,:]



