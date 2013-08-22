"""
This example explains how the Hessian of a function 
f: R^N -->R

can be combined in the combinded forward/reverse mode of AD
"""



import numpy
from algopy import CGraph, Function, UTPM, dot, qr, eigh, inv, solve, trace

# first order derivatives, one directional derivative
# D - 1 is the degree of the Taylor polynomial
# P directional derivatives at once
# M number of rows of A
# N number of cols of A
D,M,N = 2,3,3
P = M*N

# generate badly conditioned matrix A

A = UTPM(numpy.zeros((D,P,M,N)))
A.data[0,:] = numpy.random.rand(*(M,N))

for m in range(M):
    for n in range(N):
        p = m*N + n
        A.data[1,p,m,n] = 1.


cg = CGraph()
A = Function(A)
y = trace(inv(A))
cg.trace_off()

cg.independentFunctionList = [A]
cg.dependentFunctionList = [y]
    
ybar = y.x.zeros_like()
ybar.data[0,:] = 1.
cg.pullback([ybar])

# check gradient
g_forward = numpy.zeros(N*N)
g_reverse = numpy.zeros(N*N)

for m in range(M):
    for n in range(N):
        p = m*N + n
        g_forward[p] = y.x.data[1,p]
        g_reverse[p] = A.xbar.data[0,0,m,n]

numpy.testing.assert_array_almost_equal(g_forward, g_reverse)

H = numpy.zeros((M,N,M,N))
for m in range(M):
    for n in range(N):
        for m2 in range(M):
            for n2 in range(N):
                p = m2*N + n2
                H[m,n, m2,n2] = A.xbar.data[1,p,m,n]


H_reshaped =  H.reshape((M*N, M*N))

print(H_reshaped - H_reshaped.T)

# print y.x.data[1]
