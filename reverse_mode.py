#!/usr/bin/env python

from pylab import *
from numpy import *

class CGraph:
	variableVertexCount = 0
	functionVertexCount = 0
	vertexNumberCount = 0
	edgeNumberCount = 0
	vertexList = []

	class Vertex:
		def __init__(self, vertex_type):
			if vertex_type == 'v':
				self.type = 'v'
				self.vid = CGraph.variableVertexCount
				CGraph.variableVertexCount +=1
			elif vertex_type == 'add':
				self.type = 'add'
				self.fid = CGraph.functionVertexCount
				CGraph.variableVertexCount +=1
			elif vertex_type == 'mul':
				self.type = 'mul'
				self.fid = CGraph.functionVertexCount
				CGraph.variableVertexCount +=1
			else:
				raise NotImplementedError('vertex_type must be either \'v\' or \'f\' ')

			self.id = CGraph.vertexNumberCount
			CGraph.vertexNumberCount += 1
			self.out_edges = []
			self.in_edges = []

		def __str__(self):
			return 'v%s'%(str(self.id))

		def __repr__(self):
			return self.__str__()

	class Edge:
		def __init__(self,source, target):
			self.source = source
			self.target = target
			
		def __str__(self):
			return '[%s,%s]'%(str(self.source), str(self.target))
		def __repr__(self):
			return self.__str__()

class adouble:
	def __init__(self,x, dx = 0):
		self.x = x
		self.dx = dx
		self.vertex = CGraph.Vertex(vertex_type='v')
		CGraph.vertexList.append(self.vertex)
		

	def __add__(self,rhs):
		retval = adouble(self.x + rhs.x, self.dx + rhs.dx)
		
		f = CGraph.Vertex(vertex_type='add')
		e3 = CGraph.Edge(f,retval.vertex)
		retval.vertex.in_edges.append(e3)
		f.out_edges.append(e3)
		
		CGraph.vertexList.append(f)
		e1 = CGraph.Edge(self.vertex,f)
		e2 = CGraph.Edge(rhs.vertex,f)
		f.in_edges.append(e1)
		f.in_edges.append(e2)

		self.vertex.out_edges.append(e1)
		rhs.vertex.out_edges.append(e2)
		return retval

	def __mul__(self,rhs):
		retval = adouble(self.x * rhs.x, self.dx * rhs.x + self.x * rhs.dx)
		
		f = CGraph.Vertex(vertex_type='mul')
		e3 = CGraph.Edge(f,retval.vertex)
		retval.vertex.in_edges.append(e3)
		f.out_edges.append(e3)
		
		CGraph.vertexList.append(f)
		e1 = CGraph.Edge(self.vertex,f)
		e2 = CGraph.Edge(rhs.vertex,f)
		f.in_edges.append(e1)
		f.in_edges.append(e2)

		self.vertex.out_edges.append(e1)
		rhs.vertex.out_edges.append(e2)
		return retval

# tape operations
ax = adouble(1.,1.)
ay = adouble(2.)
az = ax*ay + ax*ay

# show result of the forward evaluation
print 'd/dx ( 2 * x * y) = ', az.dx

# compute reverse sweep
ubar = 1.
for v in CGraph.vertexList[-1::-1]:
	for e in v.in_edges:


from pygraphviz import *

A = AGraph()
A.node_attr['style']='filled'
A.node_attr['shape']='circle'
A.node_attr['fixedsize']='true'
A.node_attr['fontcolor']='#000000'

#for v in CGraph.vertexList:
	#for e in v.out_edges:
		#print v.id,'->',e.target.id,"  :   ",v,'->',e.target
		#A.add_edge(v.id,e.target.id)
		#s = A.get_node(v.id)
		#if v.type == 'add':
			#s.attr['fillcolor']="#000000"
			#s.attr['shape']='rect'
			#s.attr['label']='+'
			#s.attr['width']='0.3'
			#s.attr['height']='0.3'
			#s.attr['fontcolor']='#ffffff'

		#elif v.type == 'mul':
			#s.attr['fillcolor']="#000000"
			#s.attr['shape']='rect'
			#s.attr['label']='*'
			#s.attr['width']='0.3'
			#s.attr['height']='0.3'
			#s.attr['fontcolor']='#ffffff'

		#elif v.type == 'v':
			#s.attr['fillcolor']="#FFFF00"
			#s.attr['shape']='circle'
			#s.attr['label']= 'v_%d'%v.id
			#s.attr['width']='0.3'
			#s.attr['height']='0.3'

#print A.string() # print to screen

#A.write("computational_graph.dot")
#import os
#os.system("dot  computational_graph.dot -Tsvg -o computational_graph.svg")
#os.system("dot  computational_graph.dot -Tpng -o computational_graph.png")



for v in CGraph.vertexList[-1::-1]:
	for e in v.in_edges:
		print v.id,'->',e.source.id,"  :   ",v,'->',e.source
		A.add_edge(v.id,e.source.id)
		s = A.get_node(v.id)
		if v.type == 'add':
			s.attr['fillcolor']="#000000"
			s.attr['shape']='rect'
			s.attr['label']='+'
			s.attr['width']='0.3'
			s.attr['height']='0.3'
			s.attr['fontcolor']='#ffffff'

		elif v.type == 'mul':
			s.attr['fillcolor']="#000000"
			s.attr['shape']='rect'
			s.attr['label']='*'
			s.attr['width']='0.3'
			s.attr['height']='0.3'
			s.attr['fontcolor']='#ffffff'

		elif v.type == 'v':
			s.attr['fillcolor']="#FFFF00"
			s.attr['shape']='circle'
			s.attr['label']= 'v_%d'%v.id
			s.attr['width']='0.3'
			s.attr['height']='0.3'

print A.string() # print to screen

A.write("rev_computational_graph.dot")
import os
os.system("dot  rev_computational_graph.dot -Tsvg -o rev_computational_graph.svg")
os.system("dot  rev_computational_graph.dot -Tpng -o rev_computational_graph.png")



