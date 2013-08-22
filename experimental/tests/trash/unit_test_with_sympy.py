#!/usr/bin/env python
#######################################################
# This is a Unit Test t that makes use of the  Python #
# Module Sympy.                                       #
#######################################################

import sympy as sym
from numpy import array, zeros, ones, shape
from numpy.random import random
from numpy.linalg import norm
from forward_mode import *

N = 4
D = 3
M = N + 3

xs = array([[sym.Symbol('x%d%d'%(n,d)) for d in range(D)] for n in range(N)])
# computing the function f: R^(NxD) -> R symbolically
fs = 0
for n in range(1,N):
	for m in range(n):
		tmp = 0
		for d in range(D):
			tmp += (xs[n,d] - xs[m,d])**2
		tmp = sym.sqrt(tmp)
		fs += tmp

# computing the gradient symbolically
dfs = array([[sym.diff(fs, xs[n,d]) for d in range(D)] for n in range(N)])

# computing the Hessian symbolically
ddfs = array([[[[ sym.diff(dfs[m,e], xs[n,d]) for d in range(D)] for n in range(N)] for e in range(D) ] for m in range(N)])


# function f
def f(x):
	retval = 0.
	for n in range(1,N):
		for m in range(n):
			retval += 1./ norm(x[n,:] - x[m,:])
	return retval

def df(x):
	g = zeros(shape(x),dtype=float)
	for n in range(N):
		for d in range(D):
			for m in range(N):
				if n != m:
					g[n,d] += (x[n,d] - x[m,d])/norm(x[n,:]-x[m,:])
	return g

def ddf(x):
	N,D = shape(x)
	H = zeros((N,D,N,D),dtype=float)
	for n in range(N):
		for d in range(D):
			for m in range(N):
				for e in range(D):
					for l in range(N):
						if l==n:
							continue
						H[n,d,m,e] -= (( (m==n) * (d==e) - (m==l)*(d==e) ) - 3* (x[n,d] - x[l,d])/norm(x[n,:]-x[l,:])**2 * ( (n==m) - (m==l))*( x[n,e] - x[l,e]))/norm(x[n,:] - x[l,:])**3
	return H

def sym_df(x):
	symdict = dict()
	for n in range(N):
		for d in range(D):
			symdict[xs[n,d]] = x[n,d]
	return array([[dfs[n,d].subs_dict(symdict).evalf() for d in range(D)] for n in range(N)])

def sym_ddf(x):
	symdict = dict()
	for n in range(N):
		for d in range(D):
			symdict[xs[n,d]] = x[n,d]
	return array([[[[ ddfs[m,e,n,d].subs_dict(symdict).evalf() for d in range(D)] for n in range(N)] for e in range(D)] for m in range(N)],dtype=float)

def ad_df(x):
	return gradient(f,x)

def ad_ddf(x):
	return hessian(f,x)


# point at which the derivatives should be evaluated
x = random((N,D))
print('\n\n')
print('Sympy vs Hand Derived Gradient check (should be almost zero)')
print(df(x) - sym_df(x))
print('Sympy vs Ad Derive Gradient check (should be almost zero)')
print(ad_df(x) - sym_df(x))

#print '\n\n'
#print 'Hessian check (should be almost zero)'
#print ddf(x) - sym_ddf(x)





