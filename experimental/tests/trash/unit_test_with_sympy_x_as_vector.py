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

xs = array([sym.Symbol('x%d'%n) for n in range(N)])
# computing the function f: R^(NxD) -> R symbolically
fs = 0
for n in range(1,N):
	for m in range(n):
		fs += 1/(xs[n] - xs[m])

# computing the gradient symbolically
dfs = array([sym.diff(fs, xs[n]) for n in range(N)])

# computing the Hessian symbolically
ddfs = array([[ sym.diff(dfs[m], xs[n]) for n in range(N)] for m in range(N)])


def sym_f(x):
	symdict = dict()
	for n in range(N):
		symdict[xs[n]] = x[n]
	return fs.subs_dict(symdict).evalf()


def sym_df(x):
	symdict = dict()
	for n in range(N):
		symdict[xs[n]] = x[n]
	return array([dfs[n].subs_dict(symdict).evalf() for n in range(N)])

def sym_ddf(x):
	symdict = dict()
	for n in range(N):
		symdict[xs[n]] = x[n]
	return array([[ ddfs[m,n].subs_dict(symdict).evalf()  for n in range(N)]  for m in range(N)],dtype=float)


def f(x):
	retval = 0.
	for n in range(1,N):
		for m in range(n):
			retval += 1./(x[n] - x[m])
	return retval

def ad_df(x):
	return gradient(f,x)

def ad_ddf(x):
	return hessian(f,x)


# point at which the derivatives should be evaluated
x = random(N)

print('\n\n')
print('Sympy function = Ad function  check (should be almost zero)')
print(f(x) - sym_f(x))

print('\n\n')
print('Sympy vs Ad Derived Gradient check (should be almost zero)')
print(ad_df(x) - sym_df(x))

print('\n\n')
print('Sympy vs Ad Derived Hessian check (should be almost zero)')
print(sym_ddf(x) - ad_ddf(x))









