#!/usr/bin/env python
"""
Author: Sebastian Walter, HU Berlin
Package: Simple Python implementation of Algorithmic Differentiation (adouble) in forward mode

example usage: Newton's method

from forward_mode import *
import numpy

def f(x):
	return 0.5*numpy.dot(x,x)
x = numpy.array([i for i in range(100)])
g = gradient(f,x) 
print x - g # hopefully the zero vector
"""
import numpy as npy
import numpy
import networkx as nx

class adouble:
	"""Class: Algorithmic differentiation"""

	def __init__(self, *taylor_coeffs):
		"""Constructor takes a list, array, tuple and variable lenght input"""
		if not npy.isscalar(taylor_coeffs[0]):
			taylor_coeffs = npy.array(taylor_coeffs[0],dtype=float)
		self.tc = npy.array(taylor_coeffs,dtype=float)
		self.off = 0
		self.d = npy.shape(self.tc)[0]

	def get_tc(self):
		return self.tc
	def set_tc(self,x):
		self.tc[:] = x[:]

	tc = property(get_tc, set_tc)

	def copy(self):
		return adouble(self.tc)

	def __add__(self, rhs):
		"""compute new Taylorseries of the function f(x,y) = x+y, where x and y adouble objects"""
		tmp = adouble(self.tc)
		if isinstance(rhs, adouble):
			tmp.tc += rhs.tc

			return tmp
		elif npy.isscalar(rhs):
			tmp.tc[0] += rhs
			return tmp
		else:
			raise NotImplementedError

	def __radd__(self, val):
		return self+val

	def __sub__(self, rhs):
		tmp = self.copy()
		if isinstance(rhs, adouble):
			tmp.tc -= rhs.tc
			return tmp
		elif npy.isscalar(rhs):
			tmp.tc[0] -= rhs
			return tmp
		else:
			raise NotImplementedError

	def __rsub__(self, other):
		return -self + other

	def __mul__(self, rhs):
		"""compute new Taylorseries of the function f(x,y) = x*y, where x and y adouble objects"""
		if isinstance(rhs, adouble):
			return adouble(npy.array(
					[ npy.sum(self.tc[:k+1] * rhs.tc[k::-1]) for k in range(self.d)]
					))
		elif npy.isscalar(rhs):
			return adouble(rhs * self.tc)
		else:
			raise NotImplementedError("%s multiplication with adouble object" % type(rhs))

	def __rmul__(self, val):
		return self*val

	def __div__(self, rhs):
		"""compute new Taylorseries of the function f(x,y) = x/y, where x and y adouble objects"""
		if isinstance(rhs, adouble):
			y = adouble(npy.zeros(self.d))
			for k in range(self.d):
				y.tc[k] = 1./ rhs.tc[0] * ( self.tc[k] - npy.sum(y.tc[:k] * rhs.tc[k:0:-1]))
			return y
		else:
			y = adouble(npy.zeros(self.d))
			for k in range(self.d):
				y.tc[k] =  self.tc[k]/rhs
			return y

			
	def __rdiv__(self, val):
		tmp = npy.zeros(self.d)
		tmp[0] = val
		return adouble(tmp)/self

	def __neg__(self):
		return adouble(-self.tc)

	def __lt__(self,rhs):
		if npy.isscalar(rhs):
			return self.tc[0] < rhs
		return self.tc[0] < rhs.tc[0]

	def __le__(self,rhs):
		if npy.isscalar(rhs):
			return self.tc[0] <= rhs
		return self.tc[0] <= rhs.tc[0]

	def __eq__(self,rhs):
		if npy.isscalar(rhs):
			return self.tc[0] == rhs
		return self.tc[0] == rhs.tc[0]

	def __ne__(self,rhs):
		if npy.isscalar(rhs):
			return self.tc[0] != rhs
		return self.tc[0] != rhs.tc[0]

	def __ge__(self,rhs):
		if npy.isscalar(rhs):
			return self.tc[0] >= rhs
		return self.tc[0] >= rhs.tc[0]

	def __gt__(self,rhs):
		if npy.isscalar(rhs):
			return self.tc[0] > rhs
		return self.tc[0] > rhs.tc[0]

	def sqrt(self):
		y = adouble(npy.zeros(self.d))
		y.tc[0] = npy.sqrt(self.tc[0])
		for k in range(1,self.d):
			y.tc[k] = 1./(2*y.tc[0]) * ( self.tc[k] - npy.sum( y.tc[1:k] * y.tc[k-1:0:-1]))
		return y

	def __abs__(self):
		tmp = self.copy()
		if tmp.tc[0] >0:
			pass
		elif tmp.tc[0] < 0:
			tmp.tc[1:] = -tmp.tc[1:]
		else:
			raise NotImplementedError("differentiation of abs(x) at x=0")
		return tmp

	def __pow__(self, exponent):
		"""Computes the power: x^n, where n must be an int"""
		if isinstance(exponent, int):
			tmp = 1
			for i in range(exponent):
				tmp=tmp*self
			return tmp
		else:
			raise TypeError("Second argumnet must be an integer")

	def __str__(self):
		"""human readable representation of the adouble object for printing >>print adouble([1,2,3]) """
		return 'a(%s)'%str(self.tc)

	def __repr__(self):
		""" human readable output of the adouble object or debugging adouble([1,2,3]).__repr__()"""
		return 'adouble object with taylor coefficients %s'%self.tc

	
# high level functions
def gradient(f,in_x):
	if npy.isscalar(in_x) == True:
		x = array([in_x],dtype=float)
	else:
		x = in_x.copy()
	#ndim = len(numpy.shape(x))
	N = numpy.prod(numpy.shape(x))
	xshp = numpy.shape(x)
	xrshp = numpy.reshape(x,N)
	ax = numpy.array([ adouble([xrshp[n],0]) for n in range(N)])
	g = numpy.zeros(N,dtype=float)
	for n in range(N):
		ax[n].tc[1]=1.
		g[n] = f(numpy.reshape(ax,xshp)).tc[1]
		ax[n].tc[1]=0.
	return numpy.reshape(g,numpy.shape(x))


def hessian(f,in_x):
	if npy.isscalar(in_x) == True:
		x = array([in_x],dtype=float)
	else:
		x = in_x.copy()
	N = numpy.prod(numpy.shape(x))
	xshp = numpy.shape(x)
	x1D = numpy.reshape(x,N)
	ax1D = numpy.array([ adouble([x1D[n],0,0]) for n in range(N)])
	ax = numpy.reshape(ax1D,xshp)
	H = numpy.zeros((N,N),dtype=float)
	for n in range(N):
		#print 'n=',n
		ax1D[n].tc[1] = 1.
		H[n,n] = 2* f(ax).tc[2]
		ax1D[n].tc[1] = 0.
		for m in range(n+1,N):
			#print 'm=',m
			ax1D[n].tc[1]=1.
			ax1D[m].tc[1]=1.
			H[n,m] += f(ax).tc[2]
			
			ax1D[n].tc[1]=0.
			ax1D[m].tc[1]=1.
			H[n,m] -=  f(ax).tc[2]
			ax1D[n].tc[1]=1.
			ax1D[m].tc[1]=0.
			H[n,m] -=  f(ax).tc[2]
			H[m,n] = H[n,m]
			ax1D[n].tc[1] = 0.
	Hshp = list(xshp) + list(xshp)
	return numpy.reshape(H,Hshp)
