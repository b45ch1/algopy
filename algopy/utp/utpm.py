"""
Implementation of the univariate matrix polynomial.
The algebraic class is

M[t]/<t^D>

where M is the ring of matrices and t in R.

"""

import numpy.linalg
from numpy import shape, dot, zeros, ndim, asarray, sum, trace


class MatPoly:
	"""
	MatPoly == Matrix Polynomial
	This class implements univariate Taylor arithmetic on matrices, i.e.
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
	  
	  For ufuncs this arrangement is advantageous, because in this order, memory chunks of size Ndir are used and the operation on each element is the same. This is desireable to avoid cache misses.
	  See for example __mul__: there, operations of self.TC[:d+1,:,:,:]* rhs.TC[d::-1,:,:,:] has to be performed. One can see, that contiguous memory blocks are used for such operations.

	  A disadvantage of this arrangement is: it seems unnatural, it is easier to regard each direction separately.
	"""
	def __init__(self, X, Xdot = None):
		""" INPUT:	shape([X]) = (D,P,N,M)"""
		Ndim = ndim(X)
		if Ndim == 4:
			self.TC = asarray(X)
		else:
			raise NotImplementedError

	def __add__(self,rhs):
		return MatPoly(self.TC + rhs.TC)

	def __sub__(self,rhs):
		return MatPoly(self.TC - rhs.TC)
	
	def __mul__(self,rhs):
		retval = MatPoly(zeros(shape(self.TC)))
		(D,P,N,M) = shape(retval.TC)
		for d in range(D):
			retval.TC[d,:,:,:] = sum( self.TC[:d+1,:,:,:] * rhs.TC[d::-1,:,:,:], axis=0)
		return retval

	def __div__(self,rhs):
		retval = MatPoly(zeros(shape(self.TC)))
		(D,P,N,M) = shape(retval.TC)
		for d in range(D):
			retval.TC[d,:,:,:] = 1./ rhs.TC[0,:,:,:] * ( self.TC[d,:,:,:] - sum(retval.TC[:d,:,:,:] * rhs.TC[d:0:-1,:,:,:], axis=0))
		return retval

	def dot(self,rhs):
		shp = list(shape(self.TC))
		shp[3] = shape(rhs.TC)[3]
		retval = MatPoly(zeros(shp))
		(D,P,N,M) = shape(retval.TC)
		for d in range(D):
			for p in range(P):
				for c in range(d+1):
					retval.TC[d,p,:,:] += numpy.dot(self.TC[c,p,:,:], rhs.TC[d-c,p,:,:])
		return retval

	def inv(self):
		retval = MatPoly(zeros(shape(self.TC)))
		(D,P,N,M) = shape(retval.TC)
		
		# TC[0] element
		for p in range(P):
			retval.TC[0,p,:,:] = numpy.linalg.inv(self.TC[0,p,:,:])
			
		# TC[d] elements
		for d in range(1,D):
			for p in range(P):
				for c in range(1,d+1):
					retval.TC[d,p,:,:] += numpy.dot(self.TC[c,p,:,:], retval.TC[d-c,p,:,:],)
				retval.TC[d,p,:,:] =  numpy.dot(-retval.TC[0,p,:,:], retval.TC[d,p,:,:],)
		return retval
				
	def solve(self,A):
		print 'warning: this can\'t do UTPM in A, only in x'
		retval = MatPoly(zeros(shape(self.TC)))
		(D,P,N,M) = shape(retval.TC)
		assert M == 1
		for p in range(P):
			X = self.TC[:,p,:,:]
			X = numpy.reshape(X,(D,N))
			X = numpy.transpose(X)
			retval.TC[:,p,:,0] = numpy.linalg.solve(A.TC[0,0,:,:],X).T
		return retval
		

	def trace(self):
		""" returns a new MatPoly in standard format, i.e. the matrices are 1x1 matrices"""
		(D,P,N,M) = shape(self.TC)
		if N!=M:
			raise TypeError(' N == M is required')
		
		retval = zeros((D,P,1,1))
		for d in range(D):
			for p in range(P):
				retval[d,p,0,0] = trace(self.TC[d,p,:,:])
		return MatPoly(retval)

	def __getitem__(self, key):
		return MatPoly(self.TC[:,:,key[0]:key[0]+1,key[1]:key[1]+1])

	def copy(self):
		return MatPoly(self.TC.copy())

	def get_shape(self):
		return numpy.shape(self.TC[0,0,:,:])

	shape = property(get_shape)

	def get_transpose(self):
		return self.transpose()
	def set_transpose(self,x):
		raise NotImplementedError('???')
	T = property(get_transpose, set_transpose)

	def transpose(self):
		return MatPoly( numpy.transpose(self.TC,axes=(0,1,3,2)))

	def set_zero(self):
		self.TC[:,:,:,:] = 0.
		return self
	
	def __str__(self):
		return str(self.TC)

	def __repr__(self):
		return self.__str__()
 
