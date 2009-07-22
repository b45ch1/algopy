#!/usr/bin/env python

from pylab import *
from numpy import *
import numpy
import numpy.linalg
import instant


		#def middle_taylor_series(d,E):
			#"""in the case when rhs is of order E<d we have to sum over less elements"""
			#e = max(0,d-E)
			#if d>0:
				#return 	npy.sum(self.tc[e:d] * rhs.tc[d-1-e::-1], axis = 0)
			#return 0

		#for d in range(D-1,-1,-1):
			#self.tc[d] *= rhs.t0
			#self.tc[d] += middle_taylor_series(d,E)
			#if d<E:
				#self.tc[d] += self.t0*rhs.tc[d]
		#self.t0 *= rhs.t0
		#return self

c_code_adouble__imul__ = """
void mul( int tmp_lhs, double *lhs, int Ndim_lhs, int * Dims_lhs, double *lhs_tc, double rhs, int Ndim_rhs, int * Dims_rhs, double *rhs_tc){
	if(Ndim_lhs == 1){
		const int D = Dims_lhs[0];
		const int E = Dims_rhs[0];
		for(int d=D-1; d >= 0; --d){
			lhs_tc[d] *= rhs;
			const int e = (0<=d-E)*(d-E);
			for(int k = e; k < d; ++k){
				lhs_tc[d] += lhs_tc[k] * rhs_tc[d-1-e-k];
			}
			if(d<E){
				lhs_tc[d] += lhs[0] * rhs_tc[d];
			}
		}
		lhs[0]*= rhs;
	}
	else if(Ndim_lhs == 2){
		const int D = Dims_lhs[0];
		const int E = Dims_rhs[0];
		const int Ndir = Dims_lhs[1];

		for(int d=D-1; d >= 0; --d){
			for(int n = 0; n != Ndir; ++n){
				lhs_tc[d*Ndir + n] *= rhs;
			}
			const int e = (0<=d-E)*(d-E);
			for(int k = e; k < d; ++k){
				for(int n = 0; n != Ndir; ++n){
					lhs_tc[d*Ndir+n] += lhs_tc[k*Ndir + n] * rhs_tc[(d-1-e-k)*Ndir+n];
				}
			}
			if(d<E){
				for(int n = 0; n != Ndir; ++n){
					lhs_tc[d*Ndir+n] += lhs[0] * rhs_tc[d*Ndir+n];
				}
			}
		}
		lhs[0]*= rhs;
	}	
}
"""




adouble__imul__ = instant.inline_with_numpy(c_code_adouble__imul__, arrays=[['tmp_lhs','lhs'],['Ndim_lhs', 'Dims_lhs', 'lhs_tc'], ['Ndim_rhs', 'Dims_rhs', 'rhs_tc']] )


class Tc:
	"""
	Taylor coefficients Tc.
	One can propagate more than one direction at a time (called vector mode) by initializing with
	mytc = Tc(x0, [[x11,...,x1Ndir],
	               [x21,...,x2Ndir],
		            ....
		           [xD1,...,xDNdir]])
		       
	For convenience it is also possible to initialize with
	mytc = Tc([x0,x1,...,xD]) or
	mytc = Tc([[x0],[x1],...,[xD]])
	i.e. only propagating one direction at a time.
	Internally, tc is always treated as 2D-array.
	"""
	
	def __init__(self,in_t0,tc=None):
		if tc == None: # case x0 = [t0,t1,t2,...,tD]
			if ndim(in_t0) == 0:
				self.t0 = in_t0
				self.tc = array([[0.]],dtype=float)
			elif ndim(in_t0) == 1:
				self.t0 = in_t0[0]
				if size(in_t0) == 1:
					self.tc = array([[0.]],dtype=float)
				else:
					self.tc = asarray([in_t0[1:]],dtype=float).T
			elif ndim(in_t0) == 2: # case x0 = [[t0],[t1],[t2],[tD]]
				self.t0 = in_t0[0,0]
				self.tc = asarray(in_t0[1:])
			else:
				raise Exception("tc must be of the format (D,Ndir)")
			
		else:	
			self.t0 = in_t0
			if ndim(tc) == 1:
				self.tc = asarray([tc],dtype=float).T
			elif ndim(tc) == 2:
				self.tc = asarray(tc,dtype=float)
			else:
				raise Exception("tc must be of the format (D,Ndir)")
	
	def copy(self):
		return Tc(self.t0, self.tc.copy())

	# property maps
	def get_tc(self):
		return self.tc
	def set_tc(self,x):
		self.tc[:] = x[:]
	def get_t0(self):
		return self.t0
	def set_t0(self,x):
		self.t0 = x
	tc = property(get_tc, set_tc)
	t0 = property(get_t0, set_t0)

	def set_zero(self):
		self.t0 = 0.
		self.tc[:] = 0
		return self

	def resize_tc(self,rhs):
		if not isinstance(rhs,Tc):
			rhs = Tc(rhs)
		D,Ndir = shape(self.tc)
		E,Ndir2 = shape(rhs.tc)
		if Ndir < Ndir2:
			# add more directions to self
			self.tc = numpy.resize(self.tc,(D,Ndir2))
			self.tc[:,Ndir:] = 0.
			Ndir = Ndir2
			
		if D<E:
			# need to reshape self.tc now
			self.tc = numpy.resize(self.tc,(E,Ndir))
			self.tc[D:] = 0.
			D = E
		return (rhs,D,E,Ndir,Ndir2)
			

	def __iadd__(self,rhs):
		(rhs,D,E,Ndir,Ndir2) = self.resize_tc(rhs)
		self.t0 += rhs.t0
		self.tc[:E,:Ndir2] += rhs.tc[:E,:Ndir2]
		return self

	def __isub__(self,rhs):
		(rhs,D,E,Ndir,Ndir2) = self.resize_tc(rhs)
		self.t0 -= rhs.t0
		self.tc[:E,:Ndir2] -= rhs.tc[:E,:Ndir2]
		return self
	
	def __imul__(self,rhs):
		(rhs,D,E,Ndir,Ndir2) = self.resize_tc(rhs)
		lhs = numpy.array([self.t0])
		adouble__imul__(lhs, self.tc, rhs.t0, rhs.tc)
		self.t0 = lhs[0]
		#def middle_taylor_series(d,E):
			#"""in the case when rhs is of order E<d we have to sum over less elements"""
			#e = max(0,d-E)
			#if d>0:
				#return 	npy.sum(self.tc[e:d] * rhs.tc[d-1-e::-1], axis = 0)
			#return 0

		#for d in range(D-1,-1,-1):
			#self.tc[d] *= rhs.t0
			#self.tc[d] += middle_taylor_series(d,E)
			#if d<E:
				#self.tc[d] += self.t0*rhs.tc[d]
		#self.t0 *= rhs.t0
		return self

	def __idiv__(self,rhs):
		(rhs,D,E,Ndir,Ndir2) = self.resize_tc(rhs)
		self.t0 /= rhs.t0
		for d in range(D):
			e = max(0,d-E)
			self.tc[d] -= sum([self.tc[j]*rhs.tc[d-j-1] for j in range(e,d)])
			if d<E:
				self.tc[d] -=  self.t0 * rhs.tc[d]
			self.tc[d] /= rhs.t0
		return self

	def __add__(self,rhs):
		retval = self.copy()
		retval += rhs
		return retval

	def __sub__(self, rhs):
		retval = self.copy()
		retval -= rhs
		return retval
	
	def __mul__(self,rhs):
		retval = self.copy()
		retval *= rhs
		return retval
	
	def __div__(self,rhs):
		retval = self.copy()
		retval.__idiv__(rhs)
		return retval

	def __radd__(self, lhs):
		return self+lhs

	def __rsub__(self, lhs):
		return -self + lhs

	def __rmul__(self, lhs):
		return self*lhs

	def __rdiv__(self, lhs):
		lhs = Tc(lhs)
		return lhs/self
		
		

	def __neg__(self):
		retval = self.copy()
		retval.t0 = -retval.t0
		retval.tc = -retval.tc
		return retval

	def __lt__(self,rhs):
		if npy.isscalar(rhs):
			return self.t0 < rhs
		return self.t0 < rhs.t0

	def __le__(self,rhs):
		if npy.isscalar(rhs):
			return self.t0 <= rhs
		return self.t0 <= rhs.t0

	def __eq__(self,rhs):
		if npy.isscalar(rhs):
			return self.t0 == rhs
		return self.t0 == rhs.t0

	def __ne__(self,rhs):
		if npy.isscalar(rhs):
			return self.t0 != rhs
		return self.t0 != rhs.t0

	def __ge__(self,rhs):
		if npy.isscalar(rhs):
			return self.t0 >= rhs
		return self.t0 >= rhs.t0

	def __gt__(self,rhs):
		if npy.isscalar(rhs):
			return self.t0 > rhs
		return self.t0 > rhs.t0


	def sqrt(self):
		D,Ndir = shape(self.tc)
		retval = self.copy()
		retval.t0 = sqrt(self.t0)
		for d in range(0,D):
			retval.tc[d] = (self.tc[d] - sum( retval.tc[:d] * retval.tc[:d][::-1]))/(2.*retval.t0)

		return retval

	def __pow__(self, exponent):
		"""Computes the power: x^n, where n must be an int"""
		if isinstance(exponent, int):
			tmp = 1
			for i in range(exponent):
				tmp=tmp*self
			return tmp
		else:
			raise TypeError("Second argumnet must be an integer")

	def exp(self):
		D,Ndir = shape(self.tc)
		retval = self.copy()

		tmp = array([d+1. for d in range(D)]) 

		retval.t0 = exp(self.t0)
		for d in range(D):
			retval.tc[d] = (sum(self.tc[:d]*tmp[:d]*retval.tc[:d][::-1]) + retval.t0 * (d+1) * self.tc[d])/(d+1.)
		return retval

	def log(self):
		D,Ndir = shape(self.tc)
		retval = self.copy()

		tmp = array([d+1. for d in range(D)]) 
		retval.t0 = log(self.t0)
		for d in range(D):
			retval.tc[d] = ( (d+1.)* self.tc[d] - sum(self.tc[:d][::-1]*tmp[:d]*retval.tc[:d]))/((d+1.)*self.t0)
		return retval

	def sin(self):
		D,Ndir = shape(self.tc)
		retval = self.copy()
		tmpcos = self.copy()

		tmp = array([d+1. for d in range(D)]) 
		retval.t0 = sin(self.t0)
		tmpcos.t0 = cos(self.t0)

		for d in range(D):
			retval.tc[d] = ( sum(tmp[:d]*tmpcos.tc[:d]*self.tc[:d]) + (d+1.)* self.tc[d]*tmpcos.t0)/(d+1.)
			tmpcos.tc[d] = -( sum(tmp[:d]*retval.tc[:d]*self.tc[:d]) + (d+1.)* self.tc[d]*retval.t0)/(d+1.)

		return retval

	def cos(self):
		D,Ndir = shape(self.tc)
		retval = self.copy()
		tmpsin = self.copy()

		tmp = array([d+1. for d in range(D)]) 
		retval.t0 = cos(self.t0)
		tmpsin.t0 = sin(self.t0)

		for d in range(D):
			retval.tc[d] = -( sum(tmp[:d]*tmpsin.tc[:d]*self.tc[:d]) + (d+1.)* self.tc[d]*tmpsin.t0)/(d+1.)
			tmpsin.tc[d] = ( sum(tmp[:d]*retval.tc[:d]*self.tc[:d]) + (d+1.)* self.tc[d]*retval.t0)/(d+1.)
		return retval

	def __str__(self):
		return 'Tc(%s,%s)'%(str(self.t0),str(self.tc))

def tc_zeros(D,Ndir):
	return Tc(0.,numpy.zeros((D,Ndir)))

class Function:
	def __init__(self, args, function_type='var'):
		if function_type == 'var':
			self.type = 'var'
		elif function_type == 'const':
			self.type = 'const'
		elif function_type == 'id':
			self.type = 'id'
		elif function_type == 'add':
			self.type = 'add'
		elif function_type == 'sub':
			self.type = 'sub'
		elif function_type == 'mul':
			self.type = 'mul'
		elif function_type == 'div':
			self.type = 'div'	
		elif function_type == 'sqrt':
			self.type = 'sqrt'
		elif function_type == 'exp':
			self.type = 'exp'
		elif function_type == 'sin':
			self.type = 'sin'
		elif function_type == 'cos':
			self.type = 'cos'
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
			x = Tc(in_x)
			fun = Function(x, function_type='const')
			return fun
		return in_x
		
		
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

	def sqrt(self):
		return Function([self], function_type='sqrt')
	
	def exp(self):
		return Function([self], function_type='exp')
	
	def sin(self):
		return Function([self], function_type='sin')

	def cos(self):
		return Function([self], function_type='cos')	
	

	def __lt__(self,rhs):
		if npy.isscalar(rhs):
			return self.x.t0 < rhs
		return self.x.t0 < rhs.x.t0

	def __le__(self,rhs):
		if npy.isscalar(rhs):
			return self.x.t0 <= rhs
		return self.x.t0 <= rhs.x.t0

	def __eq__(self,rhs):
		if npy.isscalar(rhs):
			return self.x.t0 == rhs
		return self.x.t0 == rhs.x.t0

	def __ne__(self,rhs):
		if npy.isscalar(rhs):
			return self.x.t0 != rhs
		return self.x.t0 != rhs.x.t0

	def __ge__(self,rhs):
		if npy.isscalar(rhs):
			return self.x.t0 >= rhs
		return self.x.t0 >= rhs.x.t0

	def __gt__(self,rhs):
		if npy.isscalar(rhs):
			return self.x.t0 > rhs
		return self.x.t0 > rhs.x.t0
	
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
			return self.args

		elif self.type == 'const':
			return self.args
		
		elif self.type == 'add':
			return self.args[0].x + self.args[1].x

		elif self.type == 'sub':
			return self.args[0].x - self.args[1].x
		
		elif self.type == 'mul':
			return self.args[0].x * self.args[1].x

		elif self.type == 'div':
			return self.args[0].x.__div__(self.args[1].x)

		elif self.type == 'sqrt':
			return sqrt(self.args[0].x)

		elif self.type == 'exp':
			return exp(self.args[0].x)

		elif self.type == 'sin':
			return sin(self.args[0].x)

		elif self.type == 'cos':
			return cos(self.args[0].x)
			
		else:
			raise NotImplementedError('The operation "%s" is not supported. Please implement this case in Function.eval()!'%self.type)
		

	def reval(self):
		if self.type == 'var':
			pass

		elif self.type == 'const':
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

		elif self.type == 'sqrt':
			self.args[0].xbar += self.xbar.__div__(2*sqrt(self.args[0].x))

		elif self.type == 'exp':
			self.args[0].xbar += self.xbar * exp(self.args[0].x)
			
		elif self.type == 'sin':
			self.args[0].xbar += self.xbar * cos(self.args[0].x)
			
		elif self.type == 'cos':
			self.args[0].xbar -= self.xbar * sin(self.args[0].x)
		
		else:
			raise NotImplementedError('The operation "%s" is not supported. Please implement this case in Function.reval()!'%self.type)



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


	def plot(self, filename = None, method = None):
		"""
		accepted filenames, e.g.:
		filename = 
		'myfolder/mypic.png'
		'mypic.svg'
		etc.

		accepted methods
		method = 'dot'
		method = 'circo'
		''
		"""

		import pygraphviz
		import os

		# checking filename and converting appropriately
		if filename == None:
			filename = 'computational_graph.png'

		if method != 'dot' and method != 'circo':
			method = 'dot'
		name, extension = filename.split('.')
		if extension != 'png' and extension != 'svg':
			print 'Only *.png or *.svg are supported formats!'
			print 'Using *.png now'
			extension = 'png'

		print 'name=',name, 'extension=', extension

		# setting the style for the nodes
		A = pygraphviz.agraph.AGraph(directed=True, strict = False)
		A.node_attr['fillcolor']="#000000"
		A.node_attr['shape']='rect'
		A.node_attr['width']='0.5'
		A.node_attr['height']='0.5'
		A.node_attr['fontcolor']='#ffffff'
		A.node_attr['style']='filled'
		A.node_attr['fixedsize']='true'

		# build graph
		for f in self.functionList:
			if f.type == 'var' or f.type=='const' or f.type=='id':
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
				
			elif vtype == 'pow':
				s.attr['label']='pow%d'%nf
				
			elif vtype == 'sqrt':
				s.attr['label']='sqrt%d'%nf

			elif vtype == 'exp':
				s.attr['label']='exp%d'%nf

			elif vtype == 'sin':
				s.attr['label']='sin%d'%nf
				
			elif vtype == 'cos':
				s.attr['label']='cos%d'%nf
				
				
			elif vtype == 'id':
				s.attr['fillcolor']="#AAFF00"
				s.attr['label']= 'id_%d'%nf
				s.attr['fontcolor']='#000000'

			elif vtype == 'var':
				s.attr['fillcolor']="#FFFF00"
				s.attr['shape']='circle'
				s.attr['label']= 'v_%d'%nf
				s.attr['fontcolor']='#000000'

			elif vtype == 'const':
				s.attr['fillcolor']="#AAAAAA"
				s.attr['shape']='triangle'
				s.attr['label']= 'c_%d'%nf
				s.attr['fontcolor']='#000000'
				

				
		#print A.string() # print to screen

		A.write('%s.dot'%name)
		os.system('%s  %s.dot -T%s -o %s.%s'%(method, name, extension, name, extension))

def tape(f,in_x):
	x = in_x.copy()
	N = size(x)
	cg = CGraph()
	ax = numpy.array([Function(Tc(x[n])) for n in range(N)])
	cg.independentFunctionList = ax
	ay = f(ax)

	cg.dependentFunctionList = numpy.array([ay])
	return cg

def gradient_from_graph(cg,x=None):
	if x != None:
		cg.forward(x)
	cg.reverse(numpy.array([Tc([1.])]))
	N = size(cg.independentFunctionList)
	return numpy.array([cg.independentFunctionList[n].xbar.t0 for n in range(N)])

def gradient(f, in_x):
	cg = tape(f,in_x)
	return gradient_from_graph(cg)

def hessian(f, in_x):
	x = in_x.copy()
	N = size(x)
	cg = CGraph()
	I = eye(N)
	ax = numpy.array([Function(Tc(x[n], eye(N,1,-n).T )) for n in range(N)])
	print ax
	cg.independentFunctionList = ax
	ay = f(ax)
	cg.dependentFunctionList = numpy.array([ay])
	cg.reverse(numpy.array([Tc(1., [[0.,0.]])]))
	
	H = numpy.zeros((N,N),dtype=float)
	for r in range(N):
		H[r,:] = ax[r].xbar.tc[0,:]
			
	return H


#if __name__ == "__main__":
	#lhs = numpy.array([2.])
	#rhs = 3.
	#lhs_tc = numpy.array([1.,2.])
	#rhs_tc = numpy.array([1.,2.])
	#adouble__imul__(lhs, lhs_tc, rhs, rhs_tc)

	#print lhs
	#print lhs_tc


	#lhs = numpy.array([2.])
	#rhs = 3.
	#lhs_tc = numpy.array([[1.],[2.]])
	#rhs_tc = numpy.array([[1.],[2.]])
	#adouble__imul__(lhs, lhs_tc, rhs, rhs_tc)

	#print lhs
	#print lhs_tc
	

	## Test on Taylor Coefficients
	#x = Tc([11.,1.])
	#y = Tc([13.,1.])
	#print x+y
	#print x*y
	
	#x = Tc([11.])
	#y = Tc([13.])
	#print x+y
	#print x*y
		
		
	## Test for Reverse Mode
	
	#cg = CGraph()
	#x = Function(Tc([11.,1.]))
	#y = Function(Tc([13.,1.]))
	#z = Function(Tc([13.,1.]))

	#f = (x * y) + z*(x+y*(x*z))

	##cg.plot('trash/cg_example.svg',method='dot')

	#cg.independentFunctionList = [x,y]
	#cg.dependentFunctionList = [z]
	#cg.forward([Tc([11.,0.]), Tc([13.,1.])])
	#print z
	#cg.reverse([Tc([1.,0.])])
	#print x.xbar
	#print y.xbar

	#def f(x):
		#return 2*x[0]*x[1] + 1.
	#x = array([11.,13.])
	#cg = tape(f,x)
	#print cg
	#gradient_from_graph(cg)
	#print cg
	#print gradient(f,x)


	#N = 10
	#A = 0.125 * numpy.eye(N)
	#x = numpy.ones(N)

	#def f(x):
		#return 0.5*numpy.dot(x, numpy.dot(A,x))


	## normal function evaluation
	#y = f(x)

	## taping + reverse evaluation
	#g_reverse = gradient(f,x)

	#print g_reverse

	##taping
	#cg = tape(f,x)
	#print gradient_from_graph(cg)

	#cg.plot()
	
	
	# Test vector forward mode
	#x = Tc(3.,[[1.,1.,0.],[0.,0.,0.]])
	#y = Tc(4.,[[0.,1.,1.],[0.,0.,0.]])
	
	#print 'x+y=',x+y
	#print 'x*y=',x*y
	
	## Test hessian
	#def f(x):
		#return x[1]*x[0]
	#x = array([11.,13.])	
	#print hessian(f,x)
	
	#N = 6
	#A = 13.1234 * numpy.eye(N) + numpy.ones((N,N))
	#x = numpy.ones(N)

	#def f(x):
		#return 0.5*numpy.dot(x, numpy.dot(A,x))
	
	#print hessian(f,x)

