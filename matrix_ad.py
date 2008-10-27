#!/usr/bin/env python

from numpy import *
from numpy.linalg import *

import numpy
import numpy.linalg

class Mtc:
	def __init__(self, X, Xdot = None):
		if Xdot == None:
			self.Xdot = zeros(shape(X))
		else:
			self. Xdot = Xdot
		self.X = X

	def __add__(self,rhs):
		return Mtc(self.X + rhs.X, self.Xdot + rhs.Xdot)

	def __mul__(self,rhs):
		return Mtc( self.X * rhs.X, self.Xdot * rhs.X + self.X *  rhs.Xdot )

	def dot(self,rhs):
		return Mtc( dot(self.X, rhs.X), dot(self.Xdot, rhs.X) + dot(self.X, rhs.Xdot) )

	def inv(self):
		Y0 = numpy.linalg.inv(self.X)
		return Mtc( Y0, dot(Y0, dot(self.Xdot, Y0) ))

	def copy(self):
		return Mtc(self.X.copy(), self.Xdot.copy())

	def shape(self):
		return numpy.shape(self.X)

	def set_zero(self):
		self.X[:] = 0.
		self.Xdot[:] = 0.
		return self
	
	def __str__(self):
		return 'a{%s,\n%s}'%(str(self.X), str(self.Xdot))

	def __repr__(self):
		return self.__str__()

def inv(X):
	if X.__class__ == Mtc:
		return X.inv()
	else:
		return numpy.linalg.inv(X)

def convert(in_X):
	in_X = asarray(in_X).copy()

	# find total size
	Rb,Cb = shape(in_X)
	rows = []
	cols = []
	for r in range(Rb):
		rows.append(in_X[r,0].shape()[0])
	for c in range(Cb):
		cols.append(in_X[0,c].shape()[0])

	rowsums = [ sum(rows[:r]) for r in range(0,Rb+1)]
	colsums = [ sum(cols[:c]) for c in range(0,Cb+1)]

	X = zeros((rowsums[-1],colsums[-1]))
	Xdot = X.copy()
	
	for r in range(Rb):
		for c in range(Cb):
			X[rowsums[r]:rowsums[r+1], colsums[c]:colsums[c+1]] = in_X[r,c].X
			Xdot[rowsums[r]:rowsums[r+1], colsums[c]:colsums[c+1]] = in_X[r,c].Xdot

	return Mtc(X,Xdot)
	



class Function:
	def __init__(self, args, function_type='var'):
		if function_type == 'var':
			self.type = 'var'
		elif function_type == 'id':
			self.type = 'id'
		elif function_type == 'com':
			self.type = 'com'
		elif function_type == 'add':
			self.type = 'add'
		elif function_type == 'mul':
			self.type = 'mul'
		elif function_type == 'dot':
			self.type = 'dot'
		else:
			raise NotImplementedError('function_type must be either \'v\' or \'mul\' or  \'add\'')

		self.args = args
		self.x = self.eval()
		self.xbar = self.x.copy().set_zero()
		self.id = self.cgraph.functionCount
		self.cgraph.functionCount += 1
		self.cgraph.functionList.append(self)


	def as_function(self, in_x):
		if not isinstance(in_x, Function):
			fun = Function(self.x.copy().set_zero())
			fun.x.t0 = in_x
			return fun
		return in_x
		
		
	def __add__(self,rhs):
		rhs = self.as_function(rhs)
		return Function([self, rhs], function_type='add')

	def __mul__(self,rhs):
		rhs = self.as_function(rhs)
		return Function([self, rhs], function_type='mul')	

	def __radd__(self,lhs):
		return self + lhs

	def __rmul__(self,lhs):
		return self * lhs
	
	def __str__(self):
		try:
			ret = '%s%s:\n(x=\n%s)\n(xbar=\n%s)'%(self.type,str(self.id),str(self.x),str(self.xbar))
		except:
			ret = '%s%s:(x=%s)'%(self.type,str(self.id),str(self.x))
		return ret
	
	def __repr__(self):
		return self.__str__()

	def eval(self):
		if self.type == 'var':
			return self.args
		
		elif self.type == 'add':
			return self.args[0].x + self.args[1].x

		elif self.type == 'mul':
			return self.args[0].x * self.args[1].x

		elif self.type == 'dot':
			return numpy.dot(self.args[0].x,self.args[1].x)

	def reval(self):
		if self.type == 'var':
			pass

		elif self.type == 'add':
			self.args[0].xbar += self.xbar
			self.args[1].xbar += self.xbar

		elif self.type == 'mul':
			self.args[0].xbar += self.xbar * self.args[1].x
			self.args[1].xbar += self.xbar * self.args[0].x

		elif self.type == 'dot':
			self.args[0].xbar +=  numpy.dot(self.args[1].x, self.xbar)
			self.args[1].xbar +=  numpy.dot(self.xbar, self.args[0].x)

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
		for nf,f in enumerate(self.dependentFunctionList):
			f.xbar = xbar[nf]

		for f in self.functionList[::-1]:
			f.reval()




if __name__ == "__main__":

	### Testing Taylor series of matrices
	X = array([[1,2],[2,10]],dtype=float)
	Xdot = eye(2)
	AX = Mtc(X,Xdot)
	Y = X.copy()
	AY = Mtc(Y,eye(2))


	AW = [[AX,AY],[AY,AX]]
	print convert(AW)

	#print AX + AY
	#print AX * AY
	#print dot(AX, AY)
	#print inv(AX)
	#print inv(X)

	#### Testing Taping
	#cg = CGraph()
	#FX = Function(AX)
	#FY = Function(AY)

	#FZ = dot(FX,FY)
	#FZbar = Mtc(eye(2))
	#cg.independentFunctionList=[FX,FY]
	#cg.dependentFunctionList=[FZ]
	#cg.reverse([FZbar])

	#print cg

