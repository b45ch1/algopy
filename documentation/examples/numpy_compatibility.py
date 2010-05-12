"""
This example shows that most computations can be performed by numpy functions
on arrays of UTPM objects.

Just bear in mind that is much faster use UTPM instances of matrices than numpy.ndarrays
with UTPM elements.

"""

import numpy, os
from algopy import CGraph, Function, UTPM, dot, qr, eigh, inv

N,D,P = 2,2,1
cg = CGraph()
x = numpy.array([ Function(UTPM(numpy.random.rand(*(D,P)))) for n in range(N)])
A = numpy.outer(x,x)
A = numpy.exp(A)
y = numpy.dot(A,x)

cg.independentFunctionList = list(x)
cg.dependentFunctionList = list(y)

cg.plot(os.path.join(os.path.dirname(__file__),'numpy_dot_graph.svg'))
