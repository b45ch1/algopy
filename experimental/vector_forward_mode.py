#!/usr/bin/env python
"""
Vector Forward Mode
"""
from __future__ import division
import numpy as npy
import numpy
import instant



class adouble:
	"""Class: Algorithmic differentiation"""

	def __init__(self,  *taylor_coeffs):
		"""Constructor takes a list, array, tuple and variable lenght input"""
		if not npy.isscalar(taylor_coeffs[0]):
			taylor_coeffs = npy.array(taylor_coeffs[0],dtype=float)
		self.tc = npy.array(taylor_coeffs,dtype=float)
		self.off = 0
		self.shp = npy.shape(self.tc)
		if len(self.shp) ==2:
			self.Ndir = npy.shape(self.tc)[1]
			self.D = npy.shape(self.tc)[0]
		else:
			self.D = npy.shape(self.tc)[0]


	def get_tc(self):
		return self.tc
	def set_tc(self,x):
		self.tc[:] = x[:]

	tc = property(get_tc, set_tc)

	def copy(self):
		return adouble(self.tc)

	def __add__(self, rhs):
		"""compute new Taylorseries of the function f(x,y) = x+y, where x and y adouble objects"""
		retval = adouble(numpy.zeros(numpy.shape(self.tc)))
		if isinstance(rhs, adouble):
			adouble__add__(self.tc,rhs.tc,retval.tc)
			return retval
		elif npy.isscalar(rhs):
			retval.tc += self.tc
			retval.tc[0] += rhs
			return retval
		else:
			raise NotImplementedError

	def __radd__(self, val):
		return self+val

	def __sub__(self, rhs):
		retval = adouble(numpy.zeros(numpy.shape(self.tc)))
		if isinstance(rhs, adouble):
			adouble__sub__(self.tc,rhs.tc,retval.tc)
			return retval
		elif npy.isscalar(rhs):
			retval.tc += self.tc
			retval.tc[0] -= rhs
			return retval
		else:
			raise NotImplementedError

	def __rsub__(self, other):
		return -self + other


	def __mul__(self, rhs):
		"""compute new Taylorseries of the function f(x,y) = x*y, where x and y adouble objects"""
		if isinstance(rhs, adouble):
			retval = adouble(numpy.zeros(numpy.shape(rhs.tc)))
			adouble__mul__(self.tc,rhs.tc,retval.tc)
			return retval
			#return adouble(npy.array(
					#[ npy.sum(self.tc[:k+1] * rhs.tc[k::-1], axis = 0) for k in range(self.D)]
					#))

		elif npy.isscalar(rhs):
			return adouble(rhs * self.tc)
		else:
			raise NotImplementedError("%s multiplication with adouble object" % type(rhs))

	def __rmul__(self, val):
		return self*val

	def __div__(self, rhs):
		"""compute new Taylorseries of the function f(x,y) = x/y, where x and y adouble objects"""
		if isinstance(rhs, adouble):
			retval = numpy.zeros(numpy.shape(rhs.tc))
			adouble__div__(self.tc,rhs.tc,retval)
			return adouble(retval)
			#y = adouble(npy.zeros(self.shp))
			#for k in range(self.D):
				#y.tc[k] = 1./ rhs.tc[0] * ( self.tc[k] - npy.sum(y.tc[:k] * rhs.tc[k:0:-1], axis = 0))
			#return y
		else:
			y = adouble(npy.zeros(self.shp))
			for k in range(self.D):
				y.tc[k] =  self.tc[k]/rhs
			return y

			
	def __rdiv__(self, val):
		tmp = npy.zeros(self.shp)
		tmp[0] = val
		return adouble(tmp)/self

	def __neg__(self):
		return adouble(-self.tc)


	# TODO: the comparisons check only if any x_0 satisfies the comparison!!
	def __lt__(self,rhs):
		if npy.isscalar(rhs):
			return (self.tc[0] < rhs).any()
		return (self.tc[0] < rhs.tc[0]).any()

	def __le__(self,rhs):
		if npy.isscalar(rhs):
			return (self.tc[0] <= rhs).any()
		return (self.tc[0] <= rhs.tc[0]).any()

	def __eq__(self,rhs):
		if npy.isscalar(rhs):
			return (self.tc[0] == rhs).any()
		return (self.tc[0] == rhs.tc[0]).any()

	def __ne__(self,rhs):
		if npy.isscalar(rhs):
			return (self.tc[0] != rhs).any()
		return (self.tc[0] != rhs.tc[0]).any()

	def __ge__(self,rhs):
		if npy.isscalar(rhs):
			return (self.tc[0] >= rhs).any()
		return (self.tc[0] >= rhs.tc[0]).any()

	def __gt__(self,rhs):
		if npy.isscalar(rhs):
			return (self.tc[0] > rhs).any()
		return (self.tc[0] > rhs.tc[0]).any()

	def sqrt(self):
		y = adouble(npy.zeros(self.shp))
		y.tc[0] = npy.sqrt(self.tc[0])
		for k in range(1,self.D):
			y.tc[k] = 1./(2*y.tc[0]) * ( self.tc[k] - npy.sum( y.tc[1:k] * y.tc[k-1:0:-1]))
		return y

	def __abs__(self):
		tmp = self.copy()
		if (tmp.tc[0] >0).any():
			pass
		elif tmp.tc[0] < 0:
			tmp.tc[:] = -tmp.tc[:]
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

# helper functions for convenience
def double_to_adouble(in_x,S,D):
	"""Takes an x and a seed matrix S and returns an array of adoubles with D Taylor elements
		S is a NxP matrix, P the number of directions
	"""
	x = in_x.copy()
	N,P = numpy.shape(S)

	tmp = numpy.zeros((N,D+1,P))
	for n in range(N):
		tmp[n,0,:] = x[n]
	tmp[:,1,:] = S
	return numpy.array([ adouble(tmp[n]) for n in range(N)])


	
def generate_multi_indices(N,D):
	T = []
	def rec(r,n,N,D):
		j = r.copy()
		if n == N-1:
			j[N-1] = D - numpy.sum(j[:])
			T.append(j.copy())
			return
		for a in range( D - numpy.sum( j [:] ), -1,-1 ):
			j[n]=a
			rec(j,n+1,N,D)
	r = numpy.zeros(N,dtype=int)
	rec(r,0,N,D)
	return numpy.array(T)


def convert_multi_indices_to_pos(in_I):
	I = in_I.copy()
	M,N = numpy.shape(I)
	D = numpy.sum(I[0,:])
	retval = numpy.zeros((M,D),dtype=int)
	for m in range(M):
		i = 0
		for n in range(N):
			while I[m,n]>0:
				retval[m,i]=n
				I[m,n]-=1
				i+=1
	return retval


def generate_permutations(in_x):
	x = in_x[:]
	if len(x) <=1:
		yield x
	else:
		for perm in generate_permutations(x[1:]):
			for i in range(len(perm)+1):
				yield perm[:i] + x[0:1] + perm[i:]

# high level functions
def vector_tensor(f,in_x,D):
	def binomial(z,k):
		u = int(numpy.prod([z-i for i in range(k) ]))
		d = numpy.prod([i for i in range(1,k+1)])
		return u/d

	def gamma(i,j):
		#print 'called gamma'
		"""i and j multi-indices"""
		N = len(i)
		retval = [0.]
		def alpha(i,j,k):
			term1 = (1-2*(numpy.sum(abs(i-k))%2))
			term2 = 1
			for n in range(N):
				term2 *= binomial(i[n],k[n])
			term3 = 1
			for n in range(N):
				term3 *= binomial(D*k[n]/ numpy.sum(abs(k)), j[n] )
			term4 = (numpy.sum(abs(k))/D)**(numpy.sum(abs(i)))
			#print 'term1=', term1
			#print 'term2=', term2
			#print 'term3=', term3
			return term1*term2*term3*term4
			
		def sum_recursion(in_k, n):
			#print 'called recursion with k,n',in_k,n
			k = in_k.copy()
			if n==N:
				retval[0] += alpha(i,j,k)
				return
			for a in range(i[n]+1):
				k[n]=a
				sum_recursion(k,n+1)
		k = numpy.zeros(N,dtype=int)
		sum_recursion(k,0)
		return retval[0]

	if npy.isscalar(in_x) == True:
		x = array([in_x],dtype=float)
	else:
		x = in_x.copy()
	N = numpy.prod(numpy.shape(x))
	
	retval = numpy.zeros(tuple([N for d in range(D) ]))
	S = numpy.eye(N)
	I = generate_multi_indices(N,D)
	T = numpy.dot(I,S)
	T = T.astype(int)
	ax = double_to_adouble(x,T.T,D)
	M = numpy.shape(T)[0]
	#print M
	C = numpy.zeros((M,M))
	for m1 in range(M):
		for m2 in range(M):
			i = T[m1,:]
			j = T[m2,:]
			C[m1,m2] = gamma(i,j)
	#print C
	#print f(ax)
	a = f(ax).tc[D,:]
	#print a
	b = numpy.dot(C,a)


	J = convert_multi_indices_to_pos(I)
	retval = numpy.zeros(tuple([N for d in range(D)]))

	##case D=2
	#print 'D=',D
	#i = 0
	#for n1 in range(N):
		#for n2 in range(n1,N):
			#retval[n2,n1] = retval[n1,n2]=b[i]
			#i+=1

	for m in range(M):
		perms = []
		for p in generate_permutations(list(J[m,:])):
			perms += [p]
		print numpy.array(perms)
		for p in perms:
			retval[tuple(p)] = b[m]
		
	
	return retval


def vector_gradient(f,in_x):
	if npy.isscalar(in_x) == True:
		x = array([in_x],dtype=float)
	else:
		x = numpy.array(in_x)
	
	N = numpy.prod(numpy.shape(x))
	tmp = numpy.zeros((N,2,N))
	for n in range(N):
		tmp[n,0,:] = x[n]
	tmp[:,1,:] = numpy.eye(N)

	ax = numpy.array([ adouble(tmp[n]) for n in range(N)])
	g = f(ax)
	return g.tc[1,:]

def vector_hessian(f,in_x):
	x = in_x.copy()

	# generate directions
	N = numpy.prod(numpy.shape(x))
	M = (N*(N+1))/2
	L = (N*(N-1))/2
	S = numpy.zeros((N,M))

	s = 0
	i = 0
	for n in range(1,N+1):
		S[-n:,s:s+n] = numpy.eye(n)
		S[-n,s:s+n] = numpy.ones(n)
		s+=n
		i+=1
	S = S[::-1]
	

	# initial Taylor polynomials
	tmp = numpy.zeros((N,3,M))
	for n in range(N):
		tmp[n,0,:] = x[n]
	tmp[:,1,:] = S
	#print 'S=',S
	#print 'tmp=',tmp
	ax = numpy.array([ adouble(tmp[n]) for n in range(N)])
	#print 'ax=',ax

	h = f(ax)
	#print 'h=',h
	H = numpy.zeros((N,N))
	#print 'M,N=',M,N
	for n in range(N):
		for m in range(n):
			a =  sum(range(n+1))
			b =  sum(range(m+1))
			k =  sum(range(n+2)) - m - 1
			#print 'k,a,b=', k,a,b
			if n!=m:
				H[m,n]= H[n,m]= h.tc[2,k] - h.tc[2,a] - h.tc[2,b]
			#else:
		a =  sum(range(n+1))
		H[n,n] = h.tc[2,a]
	#print 'H=',H
	return H
	#g.tc[2,:]


	
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


if __name__ == "__main__":
	import numpy as npy
	def f(x):
		return numpy.prod(x)
	x = npy.array([i for i in range(30)])

	#compare vector with normal mode
	from time import *
	t_start = time()
	vector_hessian(f,x)
	t_end = time()
	print 'secs for vector:', t_end - t_start

	t_start = time()
	hessian(f,x)
	t_end = time()
	print 'secs for normal:', t_end - t_start
