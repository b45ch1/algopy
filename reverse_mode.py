#!/usr/bin/env python

from pylab import *
from numpy import *
import numpy

class Vertex:
	def __init__(self, x, vertex_type='id'):
		if vertex_type == 'id':
			self.type = 'id'
			self.x = x
			self.xbar = 0
		elif vertex_type == 'add':
			self.type = 'add'
			self.x = x
			self.xbar = 0
		elif vertex_type == 'mul':
			self.type = 'mul'
			self.x = x
			self.xbar = 0
		else:
			raise NotImplementedError('vertex_type must be either \'v\' or \'mul\' or  \'add\'')

		self.id = self.cgraph.vertexCount
		self.cgraph.vertexCount += 1
		self.cgraph.vertexList.append(self)

	def __add__(self,rhs):
		if not isinstance(rhs,Vertex):
			rhs = Vertex(rhs)
		retval = Vertex([self.eval(), rhs.eval()], vertex_type='add')
		Edge(self,retval,0)
		Edge(rhs,retval,1)
		return retval

	def __radd__(self,lhs):
		return self + lhs

	def __mul__(self,rhs):
		if not isinstance(rhs,Vertex):
			rhs = Vertex(rhs)
		retval = Vertex([self.eval(), rhs.eval()], vertex_type='mul')
		Edge(self,retval,0)
		Edge(rhs,retval,1)
		return retval

	def __rmul__(self,lhs):
		return self * lhs


	def eval(self):
		if self.type == 'id':
			return self.x
		elif self.type == 'add':
			return self.x[0] + self.x[1]
		elif self.type == 'mul':
			return self.x[0] * self.x[1]

	def __str__(self):
		return 'v%s'%(str(self.id))
	def __repr__(self):
		return self.__str__()





class Edge:
	def __init__(self,source, target, target_arg):
		"""Edge connects two vertices. Basically an arrow from Vertex source to Vertex target.
		target_arg specifies to which argument the edge belongs to. E.g. multiplication needs two arguments.
		In the reverse mode we need to know which edge contributed to which argument."""
		self.source = source
		self.target = target
		self.target_arg = target_arg
		self.cgraph.edgeList.append(self)

	def __str__(self):
		return '[%s,%s]'%(str(self.source), str(self.target))
	def __repr__(self):
		return self.__str__()



class CGraph:
	def __init__(self):
		self.vertexCount = 0
		self.edgeCount = 0
		self.vertexList = []
		self.edgeList = []
		self.dependentVertexList = []
		self.independentVertexList = []
		Vertex.cgraph = self
		Edge.cgraph = self

	def __str__(self):
		return 'vertices:\n' + str(CGraph.vertexList) +'\nedges:\n'+ str(CGraph.edgeList)

	def reverse_sweep(self):
		for e in self.edgeList[-1::-1]:
			s = e.source
			t = e.target
			a = e.target_arg

			if t.type == 'add':
				s.xbar += t.xbar

			if t.type == 'mul':
				s.xbar += t.xbar*t.x[int(a==0)]

			#print 't.type=',t.type
			#print '%s->%s'%(s,t)
			#print 's.xbar = %s->t.xbar%s'%(s.xbar,t.xbar)




def tape(f,in_x):
	x = in_x.copy()
	N = numpy.prod(numpy.shape(x))
	cg = CGraph()
	ax = numpy.array([Vertex(x[n]) for n in range(N)])
	cg.independentVertexList = ax
	ay = f(ax)
	cg.dependentVertexList = ay
	ay.xbar = 1.
	return cg

def gradient_from_graph(cg):
	cg.reverse_sweep()
	N = size(cg.independentVertexList)
	return numpy.array([cg.independentVertexList[n].xbar for n in range(N)])

def gradient(f, in_x):
	cg = tape(f,in_x)
	return gradient_from_graph(cg)

if __name__ == "__main__":
	import numpy
	import numpy.linalg
	
	N = 20

	A = 2.3*numpy.eye(N)
	x = numpy.ones(N)

	def f(x):
		return numpy.dot(x, numpy.dot(A,x))


	## normal function evaluation
	y = f(x)

	## taping + reverse evaluation
	g_reverse = gradient(f,x)

	## taping
	cg = tape(f,x)
	print gradient_from_graph(cg)

	print gradient(f,x)
	
	## build graph
	#a = CGraph.Vertex(2)
	#b = CGraph.Vertex(3)
	#c = CGraph.Vertex(4)
	#for i in range(10):
		#c = (a + b ) * c

	## reverse mode
	#c.xbar = 1.
	#CGraph().reverse_sweep()
	#print a.xbar
	#print b.xbar


	##print CGraph()

	#from pygraphviz import *

	#A = AGraph()
	#A.node_attr['style']='filled'
	#A.node_attr['shape']='circle'
	#A.node_attr['fixedsize']='true'
	#A.node_attr['fontcolor']='#000000'

	#for e in CGraph.edgeList:
		#A.add_edge(e.source.id, e.target.id)
		#s = A.get_node(e.source.id)
		#vtype = e.source.type
		#if vtype == 'add':
				#s.attr['fillcolor']="#000000"
				#s.attr['shape']='rect'
				#s.attr['label']='+'
				#s.attr['width']='0.3'
				#s.attr['height']='0.3'
				#s.attr['fontcolor']='#ffffff'

		#elif vtype == 'mul':
			#s.attr['fillcolor']="#000000"
			#s.attr['shape']='rect'
			#s.attr['label']='*'
			#s.attr['width']='0.3'
			#s.attr['height']='0.3'
			#s.attr['fontcolor']='#ffffff'

		#elif vtype == 'id':
			#s.attr['fillcolor']="#FFFF00"
			#s.attr['shape']='circle'
			#s.attr['label']= 'v_%d'%e.source.id
			#s.attr['width']='0.3'
			#s.attr['height']='0.3'

		#t = A.get_node(e.target.id)
		#vtype = e.target.type
		#if vtype == 'add':
				#t.attr['fillcolor']="#000000"
				#t.attr['shape']='rect'
				#t.attr['label']='+'
				#t.attr['width']='0.3'
				#t.attr['height']='0.3'
				#t.attr['fontcolor']='#ffffff'

		#elif vtype == 'mul':
			#t.attr['fillcolor']="#000000"
			#t.attr['shape']='rect'
			#t.attr['label']='*'
			#t.attr['width']='0.3'
			#t.attr['height']='0.3'
			#t.attr['fontcolor']='#ffffff'

		#elif vtype == 'id':
			#t.attr['fillcolor']="#FFFF00"
			#t.attr['shape']='circle'
			#t.attr['label']= 'v_%d'%e.target.id
			#t.attr['width']='0.3'
			#t.attr['height']='0.3'

	##print A.string() # print to screen

	#A.write("computational_graph.dot")
	#import os
	#os.system("dot  computational_graph.dot -Tsvg -o computational_graph.svg")
	#os.system("dot  computational_graph.dot -Tpng -o computational_graph.png")

