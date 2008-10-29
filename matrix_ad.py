#!/usr/bin/env python

from numpy import *
from numpy.linalg import *

import numpy
import numpy.linalg



class Tc:
	"""
	Taylor coefficients Tc.
	One can propagate more than one direction at a time (called vector mode) by initializing with
	mytc = Tc(x0, [[x11,...,x1Ndir],
	               [x21,...,x2Ndir],
		            ....
		           [xD1,...,xDNdir]])
		       
	For convenience it is also possible to initialize with
	mytc = Tc([x0,x1,...,xD])
	i.e. only propagating one direction at a time.
	"""
	def __init__(self,in_t0,tc=None):
		if tc == None:
			self.t0 = in_t0[0]
			self.tc = asarray(in_t0[1:])
		else:	
			self.t0 = in_t0
			self.tc = asarray(tc)
	
	def copy(self):
		return Tc(self.t0, self.tc.copy())
	
	def set_zero(self):
		self.t0 = 0.
		self.tc[:] = 0
		return self

	def __add__(self,rhs):
		return Tc(self.t0 + rhs.t0, self.tc[:] + rhs.tc[:])
	
	def __mul__(self,rhs):
		#print 'shape(self.tc)',shape(self.tc)
		#print 'shape(rhs.tc)',shape(rhs.tc)
		#print 'self.tc=',self.tc
		#print 'rhs.tc=',rhs.tc
		#d=2
		#print 'self.tc[:d]',self.tc[:d-1]
		#print 'rhs.tc[d-2::-1]',rhs.tc[d-2::-1]
		#print 'self.tc[:d-1] * rhs.tc[d-2::-1]', self.t0*rhs.tc[d-1] +  self.tc[:d-1] * rhs.tc[d-2::-1] + self.tc[d-1] * rhs.t0
		#exit()
		
		def middle_taylor_series(d):
			if d>1:
				return 	npy.sum(self.tc[:d-1] * rhs.tc[d-2::-1], axis = 0)
			return 0
		
		D = shape(self.tc)[0]
		E = shape(rhs.tc)[0]
		assert D==E
		return Tc(self.t0 * rhs.t0,
					 npy.array(  [ self.t0*rhs.tc[d-1] + middle_taylor_series(d)  + self.tc[d-1] * rhs.t0 for d in range(1,D+1)]
					))

	def __str__(self):
		return 'Tc(%s,%s)'%(str(self.t0),str(self.tc))



class Mtc:
	"""
	Matrix Taylor Coefficients
	This class implements Taylor arithmetic on matrices, i.e.
	[A] = \sum_{d=0} A_d t^d
	A_d = \frac{d^d}{dt^d}|_{t=0} \sum_{d=0} A_d t^d
	"""
	def __init__(self, X, Xdot = None):
		if Xdot == None:
			self.Xdot = zeros(shape(X))
		else:
			self.Xdot = asarray(Xdot)
		self.X = asarray(X)

	def __add__(self,rhs):
		return Mtc(self.X + rhs.X, self.Xdot + rhs.Xdot)

	def __mul__(self,rhs):
		return Mtc( self.X * rhs.X, self.Xdot * rhs.X + self.X *  rhs.Xdot )

	def dot(self,rhs):
		return Mtc( dot(self.X, rhs.X), dot(self.Xdot, rhs.X) + dot(self.X, rhs.Xdot) )

	def inv(self):
		Y0 = numpy.linalg.inv(self.X)
		return Mtc( Y0, dot(Y0, dot(self.Xdot, Y0) ))

	def trace(self):
		return Mtc( [[self.X.trace()]], [[self.Xdot.trace()]])

	def __getitem__(self, key):
		return Mtc(self.X[key], self.Xdot[key])

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


# Numpy wrapper
#--------------
def inv(X):
	if X.__class__ == Mtc:
		return X.inv()
	else:
		return numpy.linalg.inv(X)
#--------------


# convenience functions (should be as few as possible)
#-----------------------------------------------------
def convert(in_X):
	"""
	expects a matrix/list consisting of entries of type Mtc, e.g.
	in_X = [[Mtc1,Mtc2],[Mtc3,Mtc4]]
	and returns
	Mtc([[Mtc1.X,Mtc2.X],[Mtc3.X,Mtc4.X]], [[Mtc1.Xdot,Mtc2.Xdot],[Mtc3.Xdot,Mtc4.Xdot]])

	if a matrix/list of Function is provided, the Taylor series are extracted
	"""
	
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

	# check if input is list of Functions
	if in_X[0,0].__class__ == Function:
		for r in range(Rb):
			for c in range(Cb):
				X[rowsums[r]:rowsums[r+1], colsums[c]:colsums[c+1]] = in_X[r,c].x.X
				Xdot[rowsums[r]:rowsums[r+1], colsums[c]:colsums[c+1]] = in_X[r,c].x.Xdot

	else:
		for r in range(Rb):
			for c in range(Cb):
				X[rowsums[r]:rowsums[r+1], colsums[c]:colsums[c+1]] = in_X[r,c].X
				Xdot[rowsums[r]:rowsums[r+1], colsums[c]:colsums[c+1]] = in_X[r,c].Xdot

	return Mtc(X,Xdot)


def convert_from_tc_to_mtc(in_X):
	R,C = shape(in_X)
	X = zeros((R,C))
	Xdot = X.copy()

	for r in range(R):
		for c in range(C):
			X[r,c] = in_X[r,c].t0
			Xdot[r,c] = in_X[r,c].tc[0]
	return Mtc(X,Xdot)
#-----------------------------------------------------


class Function:
	def __init__(self, args, function_type='var'):
		if function_type == 'var':
			if type(args) == list:
				self.type = 'com'
			else:
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
		elif function_type == 'trace':
			self.type = 'trace'
		else:
			raise NotImplementedError('function_type must be either \'v\' or \'mul\' or  \'add\'')

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
			fun = Function(self.x.copy().set_zero())
			fun.x.t0 = in_x
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

	def __mul__(self,rhs):
		rhs = self.as_function(rhs)
		return Function([self, rhs], function_type='mul')	

	def __radd__(self,lhs):
		return self + lhs

	def __rmul__(self,lhs):
		return self * lhs

	def dot(self,rhs):
		rhs = self.as_function(rhs)
		return Function([self, rhs], function_type='dot')

	def trace(self):
		return Function([self], function_type='trace')

	def shape(self):
		return numpy.shape(self.x.X)
	# ----------------------------		

	# forward and reverse evaluation
	# ------------------------------
	def eval(self):
		if self.type == 'var':
			return self.args

		elif self.type == 'com':
			return convert(self.args)
		
		elif self.type == 'add':
			return self.args[0].x + self.args[1].x

		elif self.type == 'mul':
			return self.args[0].x * self.args[1].x

		elif self.type == 'dot':
			return self.args[0].x.dot(self.args[1].x)

		elif self.type == 'trace':
			return self.args[0].x.trace()

		else:
			raise Exception('Unknown function "%s". Please add rule to Mtc.eval()'%self.type)

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
			self.args[0].xbar +=  self.args[1].x.dot(self.xbar)
			self.args[1].xbar +=  self.xbar.dot(self.args[0].x)

		elif self.type == 'trace':
			N = self.args[0].x.shape()[0]
			self.args[0].xbar += Mtc( self.xbar.X[0,0]*numpy.eye(N),  self.xbar.Xdot[0,0]*numpy.eye(N))
			

		elif self.type == 'com':
			Rb,Cb = shape(self.args)
			#print 'xbar.shape()=',self.xbar.shape()
			args = asarray(self.args)
			rows = []
			cols = []
			#print type(args)
			for r in range(Rb):
				rows.append(args[r,0].shape()[0])
			for c in range(Cb):
				cols.append(args[0,c].shape()[0])

			#print rows
			#print cols

			rowsums = [ int(sum(rows[:r])) for r in range(0,Rb+1)]
			colsums = [ int(sum(cols[:c])) for c in range(0,Cb+1)]

			#print rowsums
			#print colsums
			#print 'shape of xbar=', shape(self.xbar.X)
			#print 'shape of x=', shape(self.x.X)
			
			for r in range(Rb):
				for c in range(Cb):
					#print 'args[r,c].xbar=\n',args[r,c].xbar.shape()
					#print 'rhs=\n', self.xbar[rowsums[r]:rowsums[r+1],colsums[c]:colsums[c+1]].shape()
					
					args[r,c].xbar += self.xbar[rowsums[r]:rowsums[r+1],colsums[c]:colsums[c+1]]
		
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

	def plot(self):
		import pygraphviz
		import os
		
		A = pygraphviz.agraph.AGraph(directed=True)
		A.node_attr['fillcolor']="#000000"
		A.node_attr['shape']='rect'
		A.node_attr['width']='0.5'
		A.node_attr['height']='0.5'
		A.node_attr['fontcolor']='#ffffff'
		A.node_attr['style']='filled'
		A.node_attr['fixedsize']='true'

		# build graph
		for f in self.functionList:
			if f.type == 'var':
				A.add_node(f.id)
				continue
			for a in numpy.ravel(f.args):
				A.add_edge(a.id, f.id)
				#e = A.get_edge(a.source.id, f.id)
				#e.attr['color']='green'
				#e.attr['label']='a'

		for nf,f in enumerate(self.functionList):
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
			elif vtype == 'dot':
				s.attr['label']='dot%d'%nf
			elif vtype == 'com':
				s.attr['label']='com%d'%nf
				
			elif vtype == 'trace':
				s.attr['label']='trace%d'%nf
				
		print A.string() # print to screen

		A.write("trash/computational_graph.dot")
		os.system("dot  trash/computational_graph.dot -Tsvg -o trash/computational_graph.svg")
		os.system("dot  trash/computational_graph.dot -Tpng -o trash/computational_graph.png")


if __name__ == "__main__":

	### Testing Taylor series of matrices
	X = array([[1,2],[2,10]],dtype=float)
	Xdot = eye(2)
	AX = Mtc(X,Xdot)
	Y = X.copy()
	AY = Mtc(Y,eye(2))


	AW = [[AX,AY],[AY,AX]]
	#print AW
	#print convert(AW)

	#print AX + AY
	#print AX * AY
	#print dot(AX, AY)
	#print inv(AX)
	#print inv(X)

	#### Testing Taping
	cg = CGraph()
	FX = Function(AX)
	FY = Function(AY)
	Fzer = Function(Mtc(zeros((2,2))))
	FU = FX.dot(FY)

	#FW = Function(AW)
	FV = Function([[FX,FY],[Fzer,FX]])
	FU = Function([[FX,FY],[Fzer,FX]])
	FW = FV*FU
	#print FW
	#print FV

	#FZ = dot(FX,FY)
	#FZbar = Mtc(eye(2))
	#cg.independentFunctionList=[FX,FY]
	#cg.dependentFunctionList=[FZ]
	#cg.reverse([FZbar])

	#print cg.functionList
	cg.plot()

