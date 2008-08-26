#!/usr/bin/env python

from pylab import *
from numpy import *

class CGraph:
	variableVertexCount = 0
	functionVertexCount = 0
	vertexNumberCount = 0
	edgeNumberCount = 0
	vertexList = []
	edgeList = []

	def __str__(self):
		return 'vertices:\n' + str(CGraph.vertexList) +'\nedges:\n'+ str(CGraph.edgeList)

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


		def __str__(self):
			return 'v%s'%(str(self.id))

		def __repr__(self):
			return self.__str__()

	class Edge:
		def __init__(self,source, target):
			self.source = source
			self.target = target
			CGraph.edgeList.append(self)
			
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
		CGraph.Edge(f,retval.vertex)
		
		CGraph.vertexList.append(f)
		e1 = CGraph.Edge(self.vertex,f)
		e2 = CGraph.Edge(rhs.vertex,f)

		return retval

	def __mul__(self,rhs):
		retval = adouble(self.x * rhs.x, self.dx * rhs.x + self.x * rhs.dx)
		
		f = CGraph.Vertex(vertex_type='mul')
		CGraph.Edge(f,retval.vertex)
		
		CGraph.vertexList.append(f)
		e1 = CGraph.Edge(self.vertex,f)
		e2 = CGraph.Edge(rhs.vertex,f)

		return retval

# tape operations
ax = adouble(1.,1.)
ay = adouble(2.)
az = ax*ay + ax*ay + ay

# show result of the forward evaluation
print 'd/dx ( 2 * x * y) = ', az.dx

# reverse evaluation
CGraph.edgeList[-1].target.xbar = 1.
for e in CGraph.edgeList[-1::-1]:
	if e.target.type == 'add':
		e.source.xbar = e.target.xbar
	elif e.target.type == 'mul':
		e.source.xbar = e.target.xbar


##print graph
#print CGraph()

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

	#elif vtype == 'v':
		#s.attr['fillcolor']="#FFFF00"
		#s.attr['shape']='circle'
		#s.attr['label']= 'v_%d'%e.source.id
		#s.attr['width']='0.3'
		#s.attr['height']='0.3'

#print A.string() # print to screen

#A.write("computational_graph.dot")
#import os
#os.system("dot  computational_graph.dot -Tsvg -o computational_graph.svg")
#os.system("dot  computational_graph.dot -Tpng -o computational_graph.png")

