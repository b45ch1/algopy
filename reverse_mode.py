#!/usr/bin/env python

from pylab import *
from numpy import *
import numpy

class Function:
	def __init__(self, args, function_type='var'):
		if function_type == 'var':
			self.type = 'var'
			self.args = [Arg(args, source = None)]
		elif function_type == 'id':
			self.type = 'id'
			self.args = args
		elif function_type == 'add':
			self.type = 'add'
			self.args = args
		elif function_type == 'mul':
			self.type = 'mul'
			self.args = args
		else:
			raise NotImplementedError('function_type must be either \'v\' or \'mul\' or  \'add\'')


		self.xbar = 0.
		self.id = self.cgraph.functionCount
		self.cgraph.functionCount += 1
		self.cgraph.functionList.append(self)

	def __add__(self,rhs):
		if not isinstance(rhs, Function):
			rhs = Function(rhs)
		left_adouble = Arg(self.eval(),self)
		right_adouble = Arg(rhs.eval(), rhs)
		return Function([left_adouble, right_adouble], function_type='add')

	def __mul__(self,rhs):
		if not isinstance(rhs, Function):
			rhs = Function(rhs)
		left_adouble = Arg(self.eval(),self)
		right_adouble = Arg(rhs.eval(), rhs)
		return Function([left_adouble, right_adouble], function_type='mul')

	def __radd__(self,lhs):
		return self + lhs
	
	def __rmul__(self,lhs):
		return self * lhs
	
	def eval(self):
		if self.type == 'var':
			return self.args[0].x
		elif self.type == 'id':
			return self.args[0].x
		elif self.type == 'add':
			return self.args[0].x + self.args[1].x
		elif self.type == 'mul':
			return self.args[0].x * self.args[1].x

	def reval(self):
		if self.type == 'var':
			pass
		elif self.type == 'id':
			self.args[0].source.xbar += self.xbar
		elif self.type == 'add':
			self.args[0].source.xbar += self.xbar
			self.args[1].source.xbar += self.xbar
		elif self.type == 'mul':
			self.args[0].source.xbar += self.xbar * self.args[1].x
			self.args[1].source.xbar += self.xbar * self.args[0].x

	def __str__(self):
		return '%s%s:(%f,%f)'%(self.type,str(self.id),self.eval(), self.xbar)
	def __repr__(self):
		return self.__str__()

class Arg:
	def __init__(self,x, source = None):
		self.x = x
		self.source = source

	def __str__(self):
		return '(x=%s,source=%s)'%(str(self.x), str(self.source))
	def __repr__(self):
		return self.__str__()

class CGraph:
	"""
	We implement the Computational Graph (CG) as Directed Graph
	The Graph of y = x1(x2+x3) looks like

	--- independent variables
	v1: a1(x1,None)
	v2: a2(x2,None)
	v3: a3(x3,None)

	--- function operations
	+4: [a4(v2.eval, v2), a5(v3.eval, v3)]
	*5: [a6(+4.eval,+4), a7(v1.eval,v1)]

	--- dependent variables
	v6: a8(*5.eval,*5)
	
	"""
	
	def __init__(self):
		self.argCount = 0
		self.functionCount = 0
		self.functionList = []
		self.dependentFunctionList = []
		self.independentFunctionList = []
		Arg.cgraph = self
		Function.cgraph = self

	def __str__(self):
		return 'vertices:\n' + str(self.functionList)

	def forward(self,x):
		# populate independent arguments with new values
		for nf,f in enumerate(self.independentFunctionList):
			f.args[0].x = x[nf]
			
		# traverse the computational tree
		for f in self.functionList:
			if f.type == 'var':
				continue
			for a in f.args:
				a.x = a.source.eval()

	def reverse(self,xbar):
		for nf,f in enumerate(self.dependentFunctionList):
			f.xbar = xbar[nf]

		for f in self.functionList[::-1]:
			f.reval()

	def plot(self):
		import pygraphviz
		import os
		
		A = pygraphviz.agraph.AGraph(directed=True)
		A.node_attr['fillcolor']="#000000"
		A.node_attr['shape']='rect'
		A.node_attr['width']='0.3'
		A.node_attr['height']='0.3'
		A.node_attr['fontcolor']='#ffffff'
		A.node_attr['style']='filled'
		A.node_attr['fixedsize']='true'

		# build graph
		for f in cg.functionList:
			if f.type == 'var':
				continue
			for a in f.args:
				A.add_edge(a.source.id, f.id, label='%s'%a.x)
				#e = A.get_edge(a.source.id, f.id)
				#e.attr['color']='green'
				#e.attr['label']='a'

		for nf,f in enumerate(cg.functionList):
			s = A.get_node(nf)
			vtype = f.type

			if vtype == 'add':
				s.attr['label']='+%d'%nf
			elif vtype == 'mul':
				s.attr['label']='*%d'%nf
			elif vtype == 'var':
				s.attr['fillcolor']="#FFFF00"
				s.attr['shape']='circle'
				s.attr['label']= 'v_%d'%nf
				s.attr['fontcolor']='#000000'

		print A.string() # print to screen

		A.write("trash/computational_graph.dot")
		os.system("dot  trash/computational_graph.dot -Tsvg -o trash/computational_graph.svg")
		os.system("dot  trash/computational_graph.dot -Tpng -o trash/computational_graph.png")
		



def tape(f,in_x):
	x = in_x[:]
	N = len(x)
	cg = CGraph()
	ax = numpy.array([Function(x[n]) for n in range(N)])
	cg.independentFunctionList = ax
	ay = f(ax)
	cg.dependentFunctionList = numpy.array([ay])
	return cg

def gradient_from_graph(cg,x=None):
	if x != None:
		cg.forward(x)
	cg.reverse(numpy.array([1.]))
	N = len(cg.independentFunctionList)
	return numpy.array([cg.independentFunctionList[n].xbar for n in range(N)])

def gradient(f, in_x):
	cg = tape(f,in_x)
	return gradient_from_graph(cg)


if __name__ == "__main__":
	import numpy
	import numpy.linalg
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

	# taping
	cg = tape(f,x)
	print gradient_from_graph(cg)

	cg.plot()



