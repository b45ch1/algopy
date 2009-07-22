#!/usr/bin/env python

from numpy import *
from numpy.linalg import *

import numpy
import numpy.linalg

# HIGHER ORDER DERIVATIVE TENSORS BY EXACT INTERPOLATION
# the theory is explained on page 315 of the book "Evaluating Derivatives" by Andreas Griewank,
# Chapter 13, Subsection: Multivariate Tensors via Univariate Tensors

def generate_multi_indices(N,D):
	"""
	generates 2D array of all possible multi-indices with |i| = D
	e.g.
	N=3, D=2
	array([[2, 0, 0],
       [1, 1, 0],
       [1, 0, 1],
       [0, 2, 0],
       [0, 1, 1],
       [0, 0, 2]])
	i.e. each row is one multi-index.
	"""
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


def multi_index_binomial(z,k):
	"""n and k are multi-indices, i.e.
	   n = [n1,n2,...]
	   k = [k1,k2,...]
	   and computes
	   n1!/[(n1-k1)! k1!] * n2!/[(n2-k2)! k2!] * ....
	"""
	def binomial(z,k):
		""" computes z!/[(z-k)! k!] """
		u = int(numpy.prod([z-i for i in range(k) ]))
		d = numpy.prod([i for i in range(1,k+1)])
		return u/d

	assert shape(z) == shape(k)
	N = shape(z)[0]

	return prod([ binomial(z[n],k[n]) for n in range(N)])

def multi_index_abs(z):
	return sum(z)	


def convert_multi_indices_to_pos(in_I):
	"""
	a multi-index [2,1,0] tells us that we differentiate twice w.r.t x[0] and once w.r.t
	x[1] and never w.r.t x[2]
	This multi-index represents therefore the [0,0,1] element in the derivative tensor.
	"""
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

def gamma(i,j):
	""" Compute gamma(i,j), where gamma(i,j) is define as in Griewanks book in Eqn (13.13)"""
	N = len(i)
	D = sum(j)
	retval = [0.]
		
	def binomial(z,k):
		""" computes z!/[(z-k)! k!] """
		u = int(numpy.prod([z-i for i in range(k) ]))
		d = numpy.prod([i for i in range(1,k+1)])
		return u/d
	
	def alpha(i,j,k):
		""" computes one element of the sum in the evaluation of gamma,
		i.e. the equation below 13.13 in Griewanks Book"""
		term1 = (1-2*(numpy.sum(abs(i-k))%2))
		term2 = 1
		for n in range(N):
			term2 *= binomial(i[n],k[n])
		term3 = 1
		for n in range(N):
			term3 *= binomial(D*k[n]/ numpy.sum(abs(k)), j[n] )
		term4 = (numpy.sum(abs(k))/D)**(numpy.sum(abs(i)))
		return term1*term2*term3*term4
		
	def sum_recursion(in_k, n):
		""" computes gamma(i,j).
			The summation 0<k<i, where k and i multi-indices makes it necessary to do this 
			recursively.
		"""
		k = in_k.copy()
		if n==N:
			retval[0] += alpha(i,j,k)
			return
		for a in range(i[n]+1):
			k[n]=a
			sum_recursion(k,n+1)
			
	# putting everyting together here
	k = numpy.zeros(N,dtype=int)
	sum_recursion(k,0)
	return retval[0]


# convenience functions (should be as few as possible)
#-----------------------------------------------------
def convert(in_X):
	"""
	expects an array or list consisting of entries of type Mtc, e.g.
	in_X = [[Mtc1,Mtc2],[Mtc3,Mtc4]]
	and returns
	Mtc([[Mtc1.TC,Mtc2.TC],[Mtc3.TC,Mtc4.TC]])

	if a matrix/list of Function is provided, the Taylor series are extracted
	"""
	
	in_X = numpy.array(in_X)
	Rb,Cb = numpy.shape(in_X)

	# find the degree D and number of directions P
	D = 0; 	P = 0;
	if in_X[0,0].__class__ == Function:
		for r in range(Rb):
			for c in range(Cb):
				D = max(D, in_X[r,c].x.TC.shape[0])
				P = max(P, in_X[r,c].x.TC.shape[1])		
	
	else:
		for r in range(Rb):
			for c in range(Cb):
				D = max(D, in_X[r,c].TC.shape[0])
				P = max(P, in_X[r,c].TC.shape[1])

	# find the sizes of the blocks
	rows = []
	cols = []
	for r in range(Rb):
		rows.append(in_X[r,0].shape[0])
	for c in range(Cb):
		cols.append(in_X[0,c].shape[1])
	rowsums = numpy.array([ numpy.sum(rows[:r]) for r in range(0,Rb+1)],dtype=int)
	colsums = numpy.array([ numpy.sum(cols[:c]) for c in range(0,Cb+1)],dtype=int)

	# create new matrix where the blocks will be copied into
	TC = zeros((D, P, rowsums[-1],colsums[-1]))

	# check if input is list of Functions
	if in_X[0,0].__class__ == Function:
		for r in range(Rb):
			for c in range(Cb):
				TC[:,:,rowsums[r]:rowsums[r+1], colsums[c]:colsums[c+1]] = in_X[r,c].x.TC[:,:,:,:]

	else:
		for r in range(Rb):
			for c in range(Cb):
				TC[:,:,rowsums[r]:rowsums[r+1], colsums[c]:colsums[c+1]] = in_X[r,c].TC[:,:,:,:]

	return Mtc(TC)

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
		shp = list(shape(self.TC))
		shp[3] = shape(rhs.TC)[3]
		retval = Mtc(zeros(shp))
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
		for d in range(1,D):
			for p in range(P):
				for c in range(1,d+1):
					retval.TC[d,p,:,:] += numpy.dot(self.TC[c,p,:,:], retval.TC[d-c,p,:,:],)
				retval.TC[d,p,:,:] =  numpy.dot(-retval.TC[0,p,:,:], retval.TC[d,p,:,:],)
		return retval
				
	def solve(self,A):
		print 'warning: this can\'t do UTPM in A, only in x'
		retval = Mtc(zeros(shape(self.TC)))
		(D,P,N,M) = shape(retval.TC)
		assert M == 1
		for p in range(P):
			X = self.TC[:,p,:,:]
			X = numpy.reshape(X,(D,N))
			X = numpy.transpose(X)
			retval.TC[:,p,:,0] = numpy.linalg.solve(A.TC[0,0,:,:],X).T
		return retval
		

	def trace(self):
		""" returns a new Mtc in standard format, i.e. the matrices are 1x1 matrices"""
		(D,P,N,M) = shape(self.TC)
		if N!=M:
			raise TypeError(' N == M is required')
		
		retval = zeros((D,P,1,1))
		for d in range(D):
			for p in range(P):
				retval[d,p,0,0] = trace(self.TC[d,p,:,:])
		return Mtc(retval)

	def __getitem__(self, key):
		return Mtc(self.TC[:,:,key[0]:key[0]+1,key[1]:key[1]+1])

	def copy(self):
		return Mtc(self.TC.copy())

	def get_shape(self):
		return numpy.shape(self.TC[0,0,:,:])

	shape = property(get_shape)

	def get_transpose(self):
		return self.transpose()
	def set_transpose(self,x):
		raise NotImplementedError('???')
	T = property(get_transpose, set_transpose)

	def transpose(self):
		return Mtc( numpy.transpose(self.TC,axes=(0,1,3,2)))

	def set_zero(self):
		self.TC[:,:,:,:] = 0.
		return self
	
	def __str__(self):
		return str(self.TC)

	def __repr__(self):
		return self.__str__()

class Function:
	""" Function node of the Computational Graph CGraph defined below."""
	def __init__(self, args, function_type='var'):
		if function_type == 'var':
			if type(args) == list:
				self.type = 'com'
			else:
				self.type = 'var'
		elif function_type == 'id':
			self.type = 'id'
		elif function_type == 'const':
			self.type = 'const'
		elif function_type == 'com':
			self.type = 'com'
		elif function_type == 'add':
			self.type = 'add'
		elif function_type == 'sub':
			self.type = 'sub'			
		elif function_type == 'mul':
			self.type = 'mul'
		elif function_type == 'div':
			self.type = 'div'			
		elif function_type == 'dot':
			self.type = 'dot'
		elif function_type == 'trace':
			self.type = 'trace'
		elif function_type == 'inv':
			self.type = 'inv'
		elif function_type == 'solve':
			self.type = 'solve'
		elif function_type == 'trans':
			self.type = 'trans'
		
		else:
			raise NotImplementedError('function_type "%s" is unknown, please add to Function.__init__'%function_type)

		self.args = args
		self.x = self.eval()
		self.xbar_from_x()
		self.id = self.cgraph.functionCount
		self.cgraph.functionCount += 1
		self.cgraph.functionList.append(self)


	# convenience functions
	# ---------------------
	def as_function(self, in_x):
		if not isinstance(in_x, Function):
			(D,P,N,M) = numpy.shape(self.x.TC)
			fun = Mtc(numpy.zeros([D,P]+ list(numpy.shape(in_x))))
			fun = Function(fun, function_type='const')
			for p in range(P):
				fun.x.TC[0,p,:,:] = in_x
			return fun
		return in_x

	def xbar_from_x(self):
		if type(self.x) == list:
			self.xbar = []
			for r in self.x:
				self.xbar.append([c.x.copy().set_zero() for c in r])
			return
		else:
			self.xbar = self.x.copy().set_zero()


	def __str__(self):
		try:
			ret = '%s%s:\n(x=\n%s)\n(xbar=\n%s)'%(self.type,str(self.id),str(self.x),str(self.xbar))
		except:
			ret = '%s%s:(x=%s)'%(self.type,str(self.id),str(self.x))
		return ret

	def __repr__(self):
		return self.__str__()
	# ---------------------


	# overloaded matrix operations
	# ----------------------------
	def __add__(self,rhs):
		rhs = self.as_function(rhs)
		return Function([self, rhs], function_type='add')

	def __sub__(self,rhs):
		rhs = self.as_function(rhs)
		return Function([self, rhs], function_type='sub')

	def __mul__(self,rhs):
		rhs = self.as_function(rhs)
		return Function([self, rhs], function_type='mul')

	def __div__(self,rhs):
		rhs = self.as_function(rhs)
		return Function([self, rhs], function_type='div')	

	def __radd__(self,lhs):
		return self + lhs

	def __rsub__(self,lhs):
		return -self + lhs
	
	def __rmul__(self,lhs):
		return self * lhs

	def __rdiv__(self, lhs):
		lhs = Function(Tc(lhs), function_type='const')
		return lhs/self

	def dot(self,rhs):
		rhs = self.as_function(rhs)
		return Function([self, rhs], function_type='dot')

	def trace(self):
		return Function([self], function_type='trace')

	def inv(self):
		return Function([self], function_type='inv')
	
	def get_shape(self):
		return numpy.shape(self.x)

	shape = property(get_shape)
	
	def solve(self,rhs):
		rhs = self.as_function(rhs)
		return Function([self, rhs], function_type='solve')

	def transpose(self):
		return Function([self], function_type='trans')

	def get_transpose(self):
		return self.transpose()
	def set_transpose(self,x):
		raise NotImplementedError('???')
	T = property(get_transpose, set_transpose)


	# ----------------------------		

	# forward and reverse evaluation
	# ------------------------------
	def eval(self):
		if self.type == 'var':
			return self.args
		
		elif self.type == 'const':
			return self.args

		elif self.type == 'com':
			return convert(self.args)
		
		elif self.type == 'add':
			self.args[0].xbar_from_x()
			self.args[1].xbar_from_x()
			#self.args[0].xbar.set_zero()
			#self.args[1].xbar.set_zero()
			return self.args[0].x + self.args[1].x

		elif self.type == 'sub':
			self.args[0].xbar_from_x()
			self.args[1].xbar_from_x()
			#self.args[0].xbar.set_zero()
			#self.args[1].xbar.set_zero()
			return self.args[0].x - self.args[1].x		

		elif self.type == 'mul':
			self.args[0].xbar_from_x()
			self.args[1].xbar_from_x()
			#self.args[0].xbar.set_zero()
			#self.args[1].xbar.set_zero()
			return self.args[0].x * self.args[1].x

		elif self.type == 'div':
			self.args[0].xbar_from_x()
			self.args[1].xbar_from_x()	
			#self.args[0].xbar.set_zero()
			#self.args[1].xbar.set_zero()
			return self.args[0].x.__div__(self.args[1].x)

		elif self.type == 'dot':
			self.args[0].xbar_from_x()
			self.args[1].xbar_from_x()
			#self.args[0].xbar.set_zero()
			#self.args[1].xbar.set_zero()
			return self.args[0].x.dot(self.args[1].x)

		elif self.type == 'trace':
			self.args[0].xbar_from_x()
			#self.args[0].xbar.set_zero()
			return self.args[0].x.trace()

		elif self.type == 'inv':
			self.args[0].xbar_from_x()
			#self.args[0].xbar.set_zero()
			return self.args[0].x.inv()
		
		elif self.type == 'solve':
			self.args[0].xbar_from_x()
			return self.args[0].x.solve(self.args[1].x)

		elif self.type == 'trans':
			self.args[0].xbar_from_x()
			#self.args[0].xbar.set_zero()
			return self.args[0].x.transpose()
		
		else:
			raise Exception('Unknown function "%s". Please add rule to Mtc.eval()'%self.type)

	def reval(self):
		if self.type == 'var':
			pass

		elif self.type == 'add':
			self.args[0].xbar += self.xbar
			self.args[1].xbar += self.xbar

		elif self.type == 'sub':
			self.args[0].xbar += self.xbar
			self.args[1].xbar -= self.xbar

		elif self.type == 'mul':
			self.args[0].xbar += self.xbar * self.args[1].x
			self.args[1].xbar += self.xbar * self.args[0].x

		elif self.type == 'div':
			self.args[0].xbar += self.xbar.__div__(self.args[1].x)
			self.args[1].xbar += self.xbar * self.args[0].x.__div__(self.args[1].x * self.args[1].x)

		elif self.type == 'dot':
			self.args[0].xbar +=  self.args[1].x.dot(self.xbar).T
			self.args[1].xbar +=  self.xbar.dot(self.args[0].x).T

		elif self.type == 'trace':
			(D,P,N,M) = shape( self.args[0].x.TC)
			tmp = zeros((D,P,N,M))
			for d in range(D):
				for p in range(P):
					tmp[d,p,:,:] = self.xbar.TC[d,p,0,0] * eye(N)
			self.args[0].xbar += Mtc( tmp )
			#print 'self.args[0].xbar=',self.args[0].xbar
			#exit()

		elif self.type == 'inv':
			self.args[0].xbar -= self.x.T.dot(self.xbar.dot(self.x.T))
			
		elif self.type == 'solve':
			raise NotImplementedError

		elif self.type == 'trans':
			self.args[0].xbar += self.xbar.transpose()

		elif self.type == 'com':
			Rb,Cb = shape(self.args)
			#print 'xbar.shape()=',self.xbar.shape()
			args = asarray(self.args)
			rows = []
			cols = []
			#print type(args)
			for r in range(Rb):
				rows.append(args[r,0].shape[0])
			for c in range(Cb):
				cols.append(args[0,c].shape[1])

			#print rows
			#print cols

			rowsums = [ int(sum(rows[:r])) for r in range(0,Rb+1)]
			colsums = [ int(sum(cols[:c])) for c in range(0,Cb+1)]

			#print rowsums
			#print colsums
			#print 'shape of xbar=', shape(self.xbar.TC)
			#print 'shape of x=', shape(self.x.TC)
			
			for r in range(Rb):
				for c in range(Cb):
					#print 'args[r,c].xbar=\n',args[r,c].xbar.shape()
					#print 'rhs=\n', self.xbar[rowsums[r]:rowsums[r+1],colsums[c]:colsums[c+1]].shape()
					
					args[r,c].xbar.TC[:,:,:,:] += self.xbar.TC[:,:,rowsums[r]:rowsums[r+1],colsums[c]:colsums[c+1]]
		
		else:
			raise Exception('Unknown function "%s". Please add rule to Mtc.reval()'%self.type)
			
	# ------------------------------

class CGraph:
	"""
	We implement the Computational Graph (CG) as Directed Graph
	The Graph of y = x1(x2+x3) looks like

	--- independent variables
	v1(x1): None
	v2(x2): None
	v3(x3): None

	--- function operations
	+4(v2.x + v3.x): [v2,v3] 
	*5(v1.x * +4.x): [v1,+4]

	--- dependent variables
	v6(*5.x): [*5]
	"""
	
	def __init__(self):
		self.functionCount = 0
		self.functionList = []
		self.dependentFunctionList = []
		self.independentFunctionList = []
		Function.cgraph = self

	def __str__(self):
		return 'vertices:\n' + str(self.functionList)

	def forward(self,x):
		# populate independent arguments with new values
		for nf,f in enumerate(self.independentFunctionList):
			f.args = x[nf]
			
		# traverse the computational tree
		for f in self.functionList:
			f.x = f.eval()

	def reverse(self,xbar):
		if numpy.size(self.dependentFunctionList) == 0:
			print 'You forgot to specify which variables are dependent!\n e.g. with cg.dependentFunctionList = [F1,F2]'
			return 
		
		for nf,f in enumerate(self.dependentFunctionList):
			f.xbar = xbar[nf]

		for f in self.functionList[::-1]:
			f.reval()

	def plot(self, filename = None, method = None, orientation = 'TD'):
		"""
		accepted filenames, e.g.:
		filename = 
		'myfolder/mypic.png'
		'mypic.svg'
		etc.

		accepted methods
		method = 'dot'
		method = 'circo'

		accepted orientations:
		orientation = 'LR'
		''
		"""

		import pygraphviz
		import os

		# checking filename and converting appropriately
		if filename == None:
			filename = 'computational_graph.png'

		if orientation != 'LR' and orientation != 'TD' :
			orientation = 'TD'

		if method != 'dot' and method != 'circo':
			method = 'dot'
		name, extension = filename.split('.')
		if extension != 'png' and extension != 'svg':
			print 'Only *.png or *.svg are supported formats!'
			print 'Using *.png now'
			extension = 'png'

		print 'name=',name, 'extension=', extension

		# setting the style for the nodes
		A = pygraphviz.agraph.AGraph(directed=True, strict = False, rankdir = orientation)
		A.node_attr['fillcolor']= '#ffffff'
		A.node_attr['shape']='rect'
		A.node_attr['width']='0.5'
		A.node_attr['height']='0.5'
		A.node_attr['fontcolor']="#000000"
		A.node_attr['style']='filled'
		A.node_attr['fixedsize']='true'

		# build graph

		for f in self.functionList:
			if f.type == 'var' or f.type == 'const':
				A.add_node(f.id)
				continue
			for a in numpy.ravel(f.args):
				A.add_edge(a.id, f.id)
				#e = A.get_edge(a.source.id, f.id)
				#e.attr['color']='green'
				#e.attr['label']='a'

		# extra formatting for the dependent variables
		for f in self.dependentFunctionList:
			s = A.get_node(f.id)
			s.attr['fillcolor'] = "#FFFFFF"
			s.attr['fontcolor']='#000000'

		# applying the style for the nodes
		for nf,f in enumerate(self.functionList):
			s = A.get_node(nf)
			vtype = f.type

			if vtype == 'add':
				s.attr['label']='+%d'%nf

			elif vtype == 'sub':
				s.attr['label']='-%d'%nf
				
			elif vtype == 'mul':
				s.attr['label']='*%d'%nf

			elif vtype == 'div':
				s.attr['label']='/%d'%nf
				
			elif vtype == 'var':
				s.attr['fillcolor']="#FFFFFF"
				s.attr['shape']='circle'
				s.attr['label']= 'v_%d'%nf
				s.attr['fontcolor']='#000000'
			elif vtype == 'const':
				s.attr['fillcolor']="#AAAAAA"
				s.attr['shape']='triangle'
				s.attr['label']= 'c_%d'%nf
				s.attr['fontcolor']='#000000'
				
			elif vtype == 'dot':
				s.attr['label']='dot%d'%nf
				
			elif vtype == 'com':
				s.attr['label']='com%d'%nf
				
			elif vtype == 'trace':
				s.attr['label']='tr%d'%nf

			elif vtype == 'inv':
				s.attr['label']='inv%d'%nf
				
			elif vtype == 'solve':
				s.attr['label']='slv%d'%nf

			elif vtype == 'trans':
				s.attr['label']='T%d'%nf
		#print A.string() # print to screen


		A.write('%s.dot'%name)
		os.system('%s  %s.dot -T%s -o %s.%s'%(method, name, extension, name, extension))

# Numpy wrapper
#--------------
def inv(X):
	if X.__class__ == Mtc or X.__class__ == Function:
		return X.inv()
	else:
		return numpy.linalg.inv(X)

def dot(X,Y):
	if X.__class__ == Mtc or X.__class__ == Function:
		return X.dot(Y)
	else:
		return numpy.dot(X,Y)
	
def trace(X):
	if X.__class__ == Mtc or X.__class__ == Function:
		return X.trace()
	else:
		return numpy.trace(X)
	
def solve(A,X):
	"""
	Solve the linear system AY=X
	where A (N,M) array, X (M,K) array and B (N,K) array
	"""
	if X.__class__ == Mtc or X.__class__ == Function:
		return X.solve(A)
	else:
		return numpy.solve(A,B)
