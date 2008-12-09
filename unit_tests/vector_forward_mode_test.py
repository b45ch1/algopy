#!/usr/bin/env python
import numpy
import numpy.linalg
from numpy import *

try:
	import sys
	sys.path = ['..'] + sys.path
	from vector_forward_mode import *
except:
	from vector_forward_mode import *

def almost_equal(ax,ay):
	tmp = []
	if type(ax) == type(numpy.array([])):
		axshp = numpy.shape(ax)
		ayshp = numpy.shape(ay)
		assert axshp == ayshp
		for n in range(len(ax)):
			print numpy.ravel(ax)
			tmp.append(numpy.ravel(ax)[n].tc - numpy.ravel(ay)[n].tc)
		tmp = numpy.asarray(tmp)
	else:
		tmp = ax.tc - ay.tc
			
	if numpy.sum(tmp*tmp)<10**-6:
		return True
	else:
		return False
	
def test_loaded_module():
	import vector_forward_mode
	assert False
	
def test_adouble_constructor():
	""" testing the construction with different inputs """
	a = adouble(1,2,3,4)
	b = adouble([1,2,3,4])
	flag1 =almost_equal(a,b)
	
	a = adouble([[1,1.],[2,1],[3,1]])
	b = numpy.array([[1,1.],[2,1],[3,1]])
	c = a.tc
	flag1 = ( numpy.sum((b-c)**2) < 10**-6)

def test_sum_of_squares():
	def poly(taylor_coefficients, x):
		"""compute \sum_{n=1}^N a_n x^n"""
		y = npy.array([x**i for i in range(npy.shape(taylor_coefficients)[0])])
		return npy.sum( taylor_coefficients * y)
	a = adouble(1,1,0)
	taylor_coefficients = npy.array([13.,17.,19.])
	print poly(taylor_coefficients,a)
	
def	test_simple_multipication():
	"""differentiation of f(x,y) = x*y at [5,7] in direction [13,17]"""
	def f(z):
		return z[0]*z[1]
	a = numpy.array([adouble([5.,13.]),adouble([7.,17.])])
	assert almost_equal(adouble([35. ,176.]), f(a))
	

def test_abs():
	"""differentiation of abs(x) at x=-2.3"""
	def f(x):
		return abs(x)
	a = adouble(-2.3,1.3)
	b = f(a)
	correct_result = adouble(2.3,-1.3)
	flag1 =  almost_equal(correct_result,b)
	
	a = adouble(5.1,1.)
	b = f(a)
	correct_result = adouble(5.1,1.)
	flag2 = almost_equal(correct_result,f(a))
	assert True == flag1*flag2

def test_numpy_linalg_norm():	
	print """\ndirectional derivative of norm(x) at x=[2.1,3.4] with direction d = [5.6,7.8]"""
	def f(x):
		return numpy.linalg.norm(x)
	a = numpy.array([adouble(2.1,5.6),adouble(3.4,7.8)])
	b = f(a)
	correct_result = adouble( numpy.linalg.norm([2.1,3.4]), 9.57898451145 )
	assert almost_equal(correct_result, b)

def test_sqrt():
	"""\ndirectional derivative of sqrt(x) at x = 3.1 with direction d = 7.4"""
	def f(x):
		return numpy.sqrt(x)
	a = adouble(3.1,7.4)
	b = f(a)
	correct_result = adouble( numpy.sqrt(3.1), 0.5/numpy.sqrt(3.1) * 7.4 )
	assert almost_equal(correct_result, b)

def test_mixed_arguments_double_adouble():
	"""\nfunction f(ax,y) = ax+y that works on mixed arguments, i.e. doubles (y) and adouble (x)"""
	def f(x,y):
		return x+y
	a1 = adouble(2.,13.)
	a2 = 1
	b = f(a1,a2)
	correct_result = adouble(f(a1.tc[0],a2), 13.)
	assert almost_equal(correct_result, b)
	
def test_double_mul_adouble():
	""" function f(ax,y) = ax * y"""
	def f(x,y):
		return x*y
	a1 = adouble(2.,13.)
	a2 = 5.
	b = f(a1,a2)
	print 'b=',b
	correct_result = adouble(f(a1.tc[0],a2), 65.)
	assert almost_equal(correct_result,b)

def test_multivariate_function():
	"""computing directional derivative of a function f:R^3 -> R^2 with direction d=[1,1,1]"""
	def f(x):
		return numpy.array([x[0]*x[1]*x[2], x[0]*x[0]*x[2]])
	def df(x,h):
		jac = numpy.array(	[[x[1]*x[2], 2* x[0]*x[2]],
							[x[0]*x[2],0],
							[x[0]*x[1], x[0]**2]])
		return numpy.dot(jac.T,h)
	
	a = [adouble(1.,1.),adouble(2.,1.),adouble(3.,1.)]
	b = f(a)
	h = [1,1,1]
	fa = f([1.,2.,3.])
	dfh = df([1.,2.,3.],h)
	correct_result = numpy.array(
		[adouble([fa[0],dfh[0]]),adouble([fa[1],dfh[1]])])
	assert almost_equal( correct_result,b)
	
def test_numpy_slicing():
	""" f= sum(x[1:]*x[-2::-1])"""
	def f(x):
		return numpy.sum(x[:]*x[-1::-1])
	def df(x,h):
		return numpy.dot(2*x[::-1],h)
		
	ax = numpy.array(
		 [adouble(1.,1.),adouble(2.,0),adouble(3.,0)]
		 )
	ay = f(ax)
	x = numpy.array([1,2,3])
	h = [1,0,0]
	correct_result = adouble(
		[f(x),df(x,h)]
		)
	assert almost_equal(correct_result, ay)

def test_generate_multi_indices():
	a =	numpy.array(
		[[3, 0, 0, 0],
		[2, 1, 0, 0],
		[2, 0, 1, 0],
		[2, 0, 0, 1],
		[1, 2, 0, 0],
		[1, 1, 1, 0],
		[1, 1, 0, 1],
		[1, 0, 2, 0],
		[1, 0, 1, 1],
		[1, 0, 0, 2],
		[0, 3, 0, 0],
		[0, 2, 1, 0],
		[0, 2, 0, 1],
		[0, 1, 2, 0],
		[0, 1, 1, 1],
		[0, 1, 0, 2],
		[0, 0, 3, 0],
		[0, 0, 2, 1],
		[0, 0, 1, 2],
		[0, 0, 0, 3]])
	assert numpy.sum(generate_multi_indices(4,3) -  a) < 10**-6

def test_convert_multi_indices_to_pos():
	N,D = 4,3
	I = generate_multi_indices(N,D)
	computed_pos_mat = convert_multi_indices_to_pos(I)
	true_pos_mat = numpy.array([[0, 0, 0],
       [0, 0, 1],
       [0, 0, 2],
       [0, 0, 3],
       [0, 1, 1],
       [0, 1, 2],
       [0, 1, 3],
       [0, 2, 2],
       [0, 2, 3],
       [0, 3, 3],
       [1, 1, 1],
       [1, 1, 2],
       [1, 1, 3],
       [1, 2, 2],
       [1, 2, 3],
       [1, 3, 3],
       [2, 2, 2],
       [2, 2, 3],
       [2, 3, 3],
       [3, 3, 3]], dtype=int)
	#print true_pos_mat
	#print computed_pos_mat
	assert numpy.prod(true_pos_mat == computed_pos_mat) #all entries have to be the same

def test_double_to_adouble():
	N = 4
	P = 3
	D = 2
	x = numpy.array([1.*n for n in range(N)])
	S = numpy.eye(N)[:,:P]

	ax = double_to_adouble(x,S,D)
	reprstring = r"""array([a([[ 0.  0.  0.]
 [ 1.  0.  0.]
 [ 0.  0.  0.]]),
       a([[ 1.  1.  1.]
 [ 0.  1.  0.]
 [ 0.  0.  0.]]),
       a([[ 2.  2.  2.]
 [ 0.  0.  1.]
 [ 0.  0.  0.]]),
       a([[ 3.  3.  3.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]])], dtype=object)"""
	assert ax.__repr__() == reprstring


def test_generate_permutations():
	x = [1,2,3]
	computed_perms = []
	for p in generate_permutations(x):
		computed_perms += [p]
	computed_perms = numpy.array(computed_perms)
	true_perms = numpy.array([[1, 2, 3],[2, 1, 3],[2, 3, 1],[1, 3, 2],[3, 1, 2],[3, 2, 1]],dtype=int)
	assert numpy.prod(computed_perms == true_perms)



def test_tensor():
	def f(x):
		return numpy.prod(x)
	x = numpy.array([1.,2.,3.])
	computed_tensor = vector_tensor(f,x,3)

	true_tensor = numpy.array([[[ 0.,  0.,  0.],
  [ 0.,  0.,  1.],
  [ 0.,  1.,  0.]],

 [[ 0.,  0.,  1.],
  [ 0.,  0.,  0.],
  [ 1.,  0.,  0.]],

 [[ 0.,  1.,  0.],
  [ 1.,  0.,  0.],
  [ 0.,  0.,  0.]]])
	print 'true_tensor=', true_tensor
	print 'computed_tensor=', computed_tensor
	assert numpy.prod(computed_tensor == true_tensor)

def test_vector_hessian():
	import time
	def f(x):
		return numpy.prod(x)
	x = numpy.array([i+1 for i in range(10)])
	start_time = time.time()
	computed_hessian = vector_hessian(f,x)
	end_time = time.time()
	print computed_hessian
	print 'run time=%0.6f seconds'%(end_time-start_time)
	assert False

	#def f(x):
		#rv = x.copy()
		#rv[:2] =  [0,1] + x[2:]*x[:2]
		#return rv
	#a = npy.array([adouble(1.,1.),adouble(2.,1.),adouble(3.,1.), adouble(4.,1.)])
	#print '\nf(x) = x[:2]*x[:2]=', f(a)
	
	#def f(x):
		#rv = x.copy()
		#rv[:] = [x[0]*x[1],x[0]*x[0]]
		#return rv
	#def J(x): #jacobian of the function f(x) = [ x[1]*x[2], x[1]**2]
		#rv = npy.zeros((2,2),dtype=float)
		#rv[0,0] = x[1]
		#rv[0,1] = x[0]
		#rv[1,0] = 2*x[0]
		#rv[1,1] = 0
		#return rv
	#print 'correct result=', npy.dot(J([2,3]),[3,5])
	#a = npy.array([adouble(2.,3.),adouble(3.,5.)])
	#print 'with adouble=', f(a)
	
	#a = npy.array([adouble(12.,17.),adouble(37.,19.)])
	#print map(adouble.get_tc,a)
	#print map(adouble.get_tc,a)
	#b = npy.array([adouble(112.,117.),adouble(137.,119.)])
	#print a + b
	#print map(adouble.get_tc, a+b+a)
	#print map(adouble.get_tc, a+b+a)
	
	#print 'testing sqrt:  d/dx sqrt(x)'
	#a = adouble(3.,1.)
	#print 'd/dx sqrt(x=3) =(',npy.sqrt(3),',',0.5/npy.sqrt(3),')=',npy.sqrt(a)
	
	#print 'testing scalar multiplication'
	#a = adouble(3.,1.)
	#print '(117,78)=',13 * a * a
	#print '(117,78)=',a * a * 13

	#print 'testing scalar division'
	#a = adouble(3.,1.)
	#b = adouble(4.,0.)
	#print '(0.75,0.25)', a/b
	#print '(%f,%f)='%(4./3, -4./9),b/a
	#print  '(%f,%f)='%(1./3, -1./9),1/a
	#print  '(%f,%f)='%(1.5, 0.5),a/2
	
	#print 'testing scalar addition/substraction'
	#a = adouble(3.,1.)
	#print '(6,1)=',a+3
	#print '(6,1)=',3+a
	#print '(-1,1)=',a-4
	#print '(1,-1)=',4-a


	## Linear Algebra methods
	#N = 10
	#M = 10
	#A = npy.array([[m*N + n for n in range(N)]for m in range(M)])
 	#print A
	#x = npy.array([adouble(1.,1.,0.) for n in range(N)])

	#print 'testing first derivative computation'

	#print npy.dot(A,x)

	#print 'testing second derivative computation'
	#x = npy.array([adouble(1.,0.,0.) for n in range(N)])
	#x[-1] = adouble(1.,1.,0.)

	#print '(..,..,99)=', npy.dot(x,npy.dot(A,x))

	## testing the gradient function
	#def f(x):
		#return x[0]/x[1]
	#print gradient(f, [5., 4.])

	
	#def f(x):
		#return 1/x
	#try:
		#print gradient(f, 2)
	#except:
		#pass

	## more testing of the gradient function
	#def f(x):
		#return npy.dot(x,x)
	#x = npy.array([3,4.])
	#print gradient(f,x)

	#def rosenbock_function(x,alpha=5.):
		#return alpha * (x[1] - x[0]**2)**2 +  (1. - x[0])**2
	#def rosenbock_gradient(x,alpha=5.):
		#return npy.array(	[2*alpha*(x[1] - x[0]**2)*(-2*x[0]) - 2*(1.-x[0]), 2 * alpha*(x[1]-x[0]**2)],	dtype=float)

	#print 'algopy computed gradient =',gradient(rosenbock_function,x)
	#print 'true gradient =', rosenbock_gradient(x)
	
	#def wolfe_function(x):
		#if x[0]>=abs(x[1]):
			#return 5*sqrt(9*x[0]**2 + 16*x[1]**2)
		#elif 0 < x[0] and x[0] < abs(x[1]):
			#return 9*x[0] + 16 * abs(x[1])
		#elif x[0]<=0:
			#return 9*x[0]+16*abs(x[1]) - x[0]**9

	#print 'algopy computed gradient =',gradient(wolfe_function,x)


	## testing the hessian function
	
	## on a simple function
	#A = npy.array([[11,3], [3,11]], dtype=float)
	#def f(x):
		#return 0.5* npy.dot(x, npy.dot(A,x))

	## checking the directional derivatives by hand
	##direction (1,1)
	#x = npy.array([adouble(1.,1., 0),adouble(2.,1., 0)])
	#print f(x)
	##direction (1,0)
	#x = npy.array([adouble(1.,1., 0),adouble(2.,0., 0)])
	#print f(x)
	##direction (0,1)
	#x = npy.array([adouble(1.,0., 0),adouble(2.,1., 0)])
	#print f(x)

	#x = npy.array([1.,2.])
	#H = hessian(f,x)
	#print '0 ?=',A - H
	
	## not quite so simple
	#N = 10
	#M = 10
	#A = npy.array([[m*N + n for n in range(N)]for m in range(M)])
	#A = npy.dot(A.T,A) #make it symmetric
	#x = npy.array([i for i in range(N)])
	#H = hessian(f,x)
	#print 'A-H',A-H

	#h = 0.1
	#def O(u):
		#from numpy import sum
		#return M**2*h**2 + sum(0.25*( (u[1:,1:] - u[0:-1,0:-1])**2 + (u[1:,0:-1] - u[0:-1, 1:])**2))
	#u = npy.zeros((M+1,M+1),dtype=float)
	#u[0,:]=13.
	#u[-1,:] = 11.
	#u[:,0]=17.
	#u[:,-1]=19.
	#print gradient(O,u)
	##print hessian(O,u) #this is too slow at the moment, needs 10 minutes and more!!! on a fast machine...



