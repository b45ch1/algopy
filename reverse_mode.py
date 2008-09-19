#!/usr/bin/env python

from pylab import *
from numpy import *

class CGraph:
	vertexCount = 0
	edgeCount = 0
	vertexList = []
	edgeList = []

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

			self.id = CGraph.vertexCount
			CGraph.vertexCount += 1
			CGraph.vertexList.append(self)

		def __add__(self,rhs):
			retval = CGraph.Vertex([self.eval(), rhs.eval()], vertex_type='add')
			CGraph.Edge(self,retval,0)
			CGraph.Edge(rhs,retval,1)
			return retval

		def __mul__(self,rhs):
			retval = CGraph.Vertex([self.eval(), rhs.eval()], vertex_type='mul')
			CGraph.Edge(self,retval,0)
			CGraph.Edge(rhs,retval,1)
			return retval

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

	
	def __str__(self):
		return 'vertices:\n' + str(CGraph.vertexList) +'\nedges:\n'+ str(CGraph.edgeList)

	def reverse_sweep(self):
		for e in CGraph.edgeList[-1::-1]:
			s = e.source
			t = e.target
			a = e.target_arg

			if t.type == 'add':
				s.xbar += t.xbar

			if t.type == 'mul':
				s.xbar += t.xbar*t.x[int(a==0)]

			print 't.type=',t.type
			print '%s->%s'%(s,t)
			print 's.xbar = %s->t.xbar%s'%(s.xbar,t.xbar)

			


	class Edge:
		def __init__(self,source, target, target_arg):
			"""Edge connects two vertices. Basically an arrow from Vertex source to Vertex target.
			target_arg specifies to which argument the edge belongs to. E.g. multiplication needs two arguments.
			In the reverse mode we need to know which edge contributed to which argument."""
			self.source = source
			self.target = target
			self.target_arg = target_arg
			CGraph.edgeList.append(self)
			
		def __str__(self):
			return '[%s,%s]'%(str(self.source), str(self.target))
		def __repr__(self):
			return self.__str__()


# build graph
a = CGraph.Vertex(2)
b = CGraph.Vertex(3)
c = CGraph.Vertex(4)
for i in range(10):
	c = (a + b ) * c

# reverse mode
c.xbar = 1.
CGraph().reverse_sweep()
print a.xbar
print b.xbar


#print CGraph()

from pygraphviz import *

A = AGraph()
A.node_attr['style']='filled'
A.node_attr['shape']='circle'
A.node_attr['fixedsize']='true'
A.node_attr['fontcolor']='#000000'

for e in CGraph.edgeList:
	A.add_edge(e.source.id, e.target.id)
	s = A.get_node(e.source.id)
	vtype = e.source.type
	if vtype == 'add':
			s.attr['fillcolor']="#000000"
			s.attr['shape']='rect'
			s.attr['label']='+'
			s.attr['width']='0.3'
			s.attr['height']='0.3'
			s.attr['fontcolor']='#ffffff'

	elif vtype == 'mul':
		s.attr['fillcolor']="#000000"
		s.attr['shape']='rect'
		s.attr['label']='*'
		s.attr['width']='0.3'
		s.attr['height']='0.3'
		s.attr['fontcolor']='#ffffff'

	elif vtype == 'id':
		s.attr['fillcolor']="#FFFF00"
		s.attr['shape']='circle'
		s.attr['label']= 'v_%d'%e.source.id
		s.attr['width']='0.3'
		s.attr['height']='0.3'

	t = A.get_node(e.target.id)
	vtype = e.target.type
	if vtype == 'add':
			t.attr['fillcolor']="#000000"
			t.attr['shape']='rect'
			t.attr['label']='+'
			t.attr['width']='0.3'
			t.attr['height']='0.3'
			t.attr['fontcolor']='#ffffff'

	elif vtype == 'mul':
		t.attr['fillcolor']="#000000"
		t.attr['shape']='rect'
		t.attr['label']='*'
		t.attr['width']='0.3'
		t.attr['height']='0.3'
		t.attr['fontcolor']='#ffffff'

	elif vtype == 'id':
		t.attr['fillcolor']="#FFFF00"
		t.attr['shape']='circle'
		t.attr['label']= 'v_%d'%e.target.id
		t.attr['width']='0.3'
		t.attr['height']='0.3'

#print A.string() # print to screen

A.write("computational_graph.dot")
import os
os.system("dot  computational_graph.dot -Tsvg -o computational_graph.svg")
os.system("dot  computational_graph.dot -Tpng -o computational_graph.png")

