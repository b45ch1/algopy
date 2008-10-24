#!/usr/bin/env python

from pylab import *
from numpy import *
import numpy

class Tc:
	def __init__(self,tc):
		self.tc = asarray(tc)

	def __add__(self,rhs):
		return Tc(self.tc[:] + rhs.tc[:])
	
	def __mul__(self,rhs):
		D = size(self.tc)
		E = size(rhs.tc)
		assert D==E
		assert size(self.tc)>0
		assert size(rhs.tc)>0

		return Tc(npy.array(
					[ npy.sum(self.tc[:k+1] * rhs.tc[k::-1], axis = 0) for k in range(D)]
					))


	def __str__(self):
		return str(self.tc)

class Function:
	def __init__(self, args, function_type='var'):
		if function_type == 'var':
			self.type = 'var'
		elif function_type == 'id':
			self.type = 'id'
		elif function_type == 'add':
			self.type = 'add'
		elif function_type == 'mul':
			self.type = 'mul'
		else:
			raise NotImplementedError('function_type must be either \'v\' or \'mul\' or  \'add\'')

		self.args = args
		self.x = self.eval()
		self.xbar = Tc(zeros(size(self.x)))
		self.id = self.cgraph.functionCount
		self.cgraph.functionCount += 1
		self.cgraph.functionList.append(self)

	def __add__(self,rhs):
		if not isinstance(rhs, Function):
			a = zeros(size(self.x.tc))
			a[0] = rhs
			rhs = Function(a)
		return Function([self, rhs], function_type='add')

	def __mul__(self,rhs):
		if not isinstance(rhs, Function):
			a = zeros(size(self.x.tc))
			a[0] = rhs
			rhs = Function(a)
		return Function([self, rhs], function_type='mul')	

	def __radd__(self,lhs):
		return self + lhs

	def __rmul__(self,lhs):
		return self * lhs
	
	def __str__(self):
		try:
			ret = '%s%s:(x=%s)(xbar=%s)'%(self.type,str(self.id),str(self.x),str(self.xbar))
		except:
			ret = '%s%s:(x=%s)'%(self.type,str(self.id),str(self.x))
		return ret
	
	def __repr__(self):
		return self.__str__()

	def eval(self):
		if self.type == 'var':
			return Tc(self.args)
		
		elif self.type == 'add':
			return self.args[0].x + self.args[1].x

		elif self.type == 'mul':
			return self.args[0].x * self.args[1].x

	def reval(self):
		if self.type == 'var':
			pass


		elif self.type == 'add':
			self.args[0].xbar += self.xbar
			self.args[1].xbar += self.xbar

		elif self.type == 'mul':
			self.args[0].xbar += self.xbar * self.args[1].x
			self.args[1].xbar += self.xbar * self.args[0].x


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

def tape(f,in_x):
	x = in_x.copy()
	N = size(x)
	cg = CGraph()
	ax = numpy.array([Function([x[n]]) for n in range(N)])
	cg.independentFunctionList = ax
	ay = f(ax)
	cg.dependentFunctionList = numpy.array([ay])
	return cg

def gradient_from_graph(cg,x=None):
	if x != None:
		cg.forward(x)
	cg.reverse(numpy.array([Tc([1.])]))
	N = size(cg.independentFunctionList)
	return numpy.array([cg.independentFunctionList[n].xbar.tc[0] for n in range(N)])

def gradient(f, in_x):
	cg = tape(f,in_x)
	return gradient_from_graph(cg)


if __name__ == "__main__":
	cg = CGraph()
	x = Function([11.,0])
	y = Function([13.,0.])
	z = x*y
	cg.independentFunctionList = [x,y]
	cg.dependentFunctionList = [z]
	cg.forward([[11.,0.],[13.,1.]])
	print z
	cg.reverse([Tc([1.,0.])])
	print x.xbar
	print y.xbar

	def f(x):
		return x[0]*x[1] + 2323
	x = array([11.,13.])
	cg = tape(f,x)
	print cg
	gradient_from_graph(cg)
	#print cg
	print gradient(f,x)


	N = 10
	A = 0.125 * numpy.eye(N)
	x = numpy.ones(N)

	def f(x):
		return 0.5*numpy.dot(x, numpy.dot(A,x))


	# normal function evaluation
	y = f(x)

	# taping + reverse evaluation
	g_reverse = gradient(f,x)

	print g_reverse

	#taping
	cg = tape(f,x)
	print gradient_from_graph(cg)

	##cg.plot()



