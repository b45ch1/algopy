#!/usr/bin/env python

from numpy import *
from numpy.linalg import *

import numpy
import numpy.linalg

class Mtc:
	"""
	Matrix Taylor Coefficients
	This class implements Taylor arithmetic on matrices, i.e.
	[A] = \sum_{d=0}^D A_d t^d
	A_d = \frac{d^d}{dt^d}|_{t=0} \sum_{c=0}^D A_c t^c
	
	in vector forward mode
	Input: 
	in the most general form, the input is a 4-tensor.
	We use the notation: 
	P: number of directions
	D: degree of the Taylor series
	N: number of rows of A_0
	M: number of cols of A_0
	
	shape([A]) = (D,P,N,M)
	The reason for this choice is that the (N,M) matrix is the elementary type, so that memory should be contiguous. Then, at each operation, the code performed to compute 
	v_d has to be repeated for every direction.
	E.g. a multiplication
	[w] = [u]*[v] =
	[[u_11, ..., u_1Ndir],
	 ...
	 [u_D1, ..., u_DNdir]]  +
	[[v11, ..., v_1Ndir],
	 ...
	 [v_D1, ..., v_DNdir]] =
	 [[ u_11 + v_11, ..., u_1Ndir + v_1Ndir],
	 ...
	  [[ u_D1 + v_D1, ..., u_DNdir + v_DNdir]]
	  One can see, that in this order, memory chunks of size Ndir are used and the operation on each element is the same. This is desireable to avoid cache misses.
	"""
	def __init__(self, X, Xdot = None):
		""" INPUT:	shape([X]) = (D,P,N,M)"""
		Ndim = ndim(X)
		if Ndim == 4:
			self.TC = asarray(X)


	def __add__(self,rhs):
		return Mtc(self.TC + rhs.TC)

	def __sub__(self,rhs):
		return Mtc(self.TC - rhs.TC)
	
	def __mul__(self,rhs):
		retval = Mtc(zeros(shape(self.TC)))
		(D,P,N,M) = shape(retval.TC)
		for d in range(D):
			retval.TC[d,:,:,:] = sum( self.TC[:d+1,:,:,:] * rhs.TC[d::-1,:,:,:], axis=0)
		return retval

	def __div__(self,rhs):
		retval = Mtc(zeros(shape(self.TC)))
		(D,P,N,M) = shape(retval.TC)
		for d in range(D):
			retval.TC[d,:,:,:] = 1./ rhs.TC[0,:,:,:] * ( self.TC[d,:,:,:] - sum(retval.TC[:d,:,:,:] * rhs.TC[d:0:-1,:,:,:], axis=0))
		return retval

	def dot(self,rhs):
		retval = Mtc(zeros(shape(self.TC)))
		(D,P,N,M) = shape(retval.TC)
		for d in range(D):
			for p in range(P):
				for c in range(d+1):
					retval.TC[d,p,:,:] += numpy.dot(self.TC[c,p,:,:], rhs.TC[d-c,p,:,:])
		return retval

	def inv(self):
		retval = Mtc(zeros(shape(self.TC)))
		(D,P,N,M) = shape(retval.TC)
		
		# TC[0] element
		for p in range(P):
			retval.TC[0,p,:,:] = numpy.linalg.inv(self.TC[0,p,:,:])
			
		# TC[d] elements
		for d in range(D):
			for p in range(P):
				for c in range(d):
					retval.TC[d,p,:,:] += numpy.dot(retval.TC[c,p,:,:], self.TC[d-c,p,:,:])
					retval.TC[d,p,:,:] = - numpy.dot(self.TC[0,p,:,:], retval.TC[d,p,:,:])
		return retval

	#def trace(self):
		#return Mtc( [[self.X.trace()]], [[self.Xdot.trace()]])

	#def __getitem__(self, key):
		#return Mtc(self.X[key], self.Xdot[key])

	#def copy(self):
		#return Mtc(self.X.copy(), self.Xdot.copy())

	#def shape(self):
		#return numpy.shape(self.X)

	#def get_transpose(self):
		#return self.transpose()
	#def set_transpose(self,x):
		#raise NotImplementedError('???')
	#T = property(get_transpose, set_transpose)

	#def transpose(self):
		#return Mtc(self.X.transpose(), self.Xdot.transpose())

	#def set_zero(self):
		#self.X[:] = 0.
		#self.Xdot[:] = 0.
		#return self
	
	def __str__(self):
		return str(self.TC)

	def __repr__(self):
		return self.__str__()
		
if __name__ == "__main__":
	import numpy.random
	X = 2 * numpy.random.rand(2,2,2,2)
	Y = 3 * numpy.random.rand(2,2,2,2)

	AX = Mtc(X)
	AY = Mtc(Y)
	AZ = AX + AY
	AZ = AX - AY
	AZ = AX * AY
	AZ = AX / AY
	AZ = AX.dot(AY)
	AZ = AX.inv()
	print 'AX=',AX
	print 'AY=',AY
	print 'AZ=',AZ
