#!/usr/bin/env python

from numpy import *
from sympy import *

N = 3
x = [Symbol('x%d'%n) for n in range(N)]

# define function f: R^N -> R
f = 1
for n in range(N):
	f *= x[n]
print 'function=',f

# compute gradient
g = array([ diff(f,x[n]) for n in range(N)])
print 'gradient=',g

# compute Hessian
H = array([[diff(g[m],x[n]) for m in range(N)] for n in range(N)])

print 'Hessian=\n',H


# evaluating the Hessian at x = [0,1,2,3,...,N-1]
symdict = dict()
for n in range(N):
	symdict[x[n]] = n
H1 = [[H[n,m].subs_dict(symdict).evalf() for n in range(N)] for m in range(N)]

print H1