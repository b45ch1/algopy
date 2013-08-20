#!/usr/bin/env python

try:
	from reverse_mode import *
except:
	import sys
	sys.path = ['..'] + sys.path
	from reverse_mode import *

#############################################################
#   TESTING CLASS TC
#############################################################
# TESTING ALL FUNCTIONS FOR BASIC FUNCTIONALITY


# testing the __init_function
def test_constructor_single_direction_list_as_input():
	inputlist = [3.]
	a = Tc(inputlist)
	assert a.t0 == 3.
	assert numpy.prod(a.tc[:] == numpy.array([inputlist[1:]]).T)
	
	inputlist = [3.,1.,2.]
	a = Tc(inputlist)
	assert a.t0 == 3.
	assert numpy.prod(a.tc[:] == numpy.array([inputlist[1:]]).T)

def test_constructor_single_direction_array_as_input():
	inputarray = numpy.array([3.,1.,2.])
	a = Tc(inputarray)
	assert a.t0 == 3.
	assert numpy.prod(a.tc[:] == array([inputarray[1:]]).T)

def test_constructor_single_direction_variable_input_length():
	a = Tc(3.)
	assert a.t0 == 3.
	assert numpy.prod(a.tc[:] == numpy.array([[]]).T)

	# Todo: variable inputlength!
	assert True


# incremental operators
def test_incremental_addition_single_direction_Tc_Tc_same_order():
	D = 4
	inputarray1 = numpy.array([[1.* i for i in range(D)]]).T
	inputarray2 = numpy.array([[1. +  i for i in range(D)]]).T
	inputarray3 = inputarray1.copy() #need to copy since Tc constructor does not copy the memory

	a = Tc(inputarray1)
	b = Tc(inputarray2)

	a += b
	assert a.t0 == 1.
	assert numpy.prod(a.tc == (inputarray3[1:] + inputarray2[1:]))

def test_incremental_addition_multiple_directions_with_constant():
	D = 4
	t0 = 2.
	tc = array([[3,5,7],[23,43,45]])
	a = Tc(t0,tc)
	a += 2.
	assert a.t0 == 4.
	assert numpy.prod(a.tc == tc)
	
def test_incremental_addition_single_direction_Tc_Tc_different_order():
	D = 4
	E = 7
	G = min(D,E)
	inputarray1 = numpy.array([[1.* i for i in range(D)]]).T
	inputarray2 = numpy.array([[1. +  i for i in range(E)]]).T
	inputarray3 = inputarray1.copy() #need to copy since Tc constructor does not copy the memory

	a = Tc(inputarray1)
	b = Tc(inputarray2)

	a += b
	assert a.t0 == 1.
	assert numpy.prod(a.tc[:G-1] == (inputarray3[1:G] + inputarray2[1:G]))
	assert numpy.prod(a.tc[G-1:] == inputarray2[G:])


def test_incremental_addition_multiple_directions_Tc_Tc_same_order():
	D = 4
	Ndir = 3
	inputarray1 = numpy.array([[1.* i + D*j for i in range(D)] for j in range(Ndir) ]).T
	inputarray2 = numpy.array([[1. +  i + D*j for i in range(D)] for j in range(Ndir)]).T
	inputarray3 = inputarray1.copy() #need to copy since Tc constructor does not copy the memory

	a = Tc(3.,inputarray1)
	b = Tc(7.,inputarray2)

	a += b

	print('inputarray3=\n',inputarray3)
	print('inputarray2=\n',inputarray2)
	print('inputarray3[1:] + inputarray2[1:]=\n',inputarray3[:] + inputarray2[:])
	assert a.t0 == 10.
	assert numpy.prod(a.tc == (inputarray3[:] + inputarray2[:]))

def test_incremental_substraction_multiple_directions_Tc_Tc_different_order():
	D = 4
	E = 7
	G = min(D,E)
	Ndir = 3
	inputarray1 = numpy.array([[1.* i + D*j for i in range(D)] for j in range(Ndir) ]).T
	inputarray2 = numpy.array([[1. +  i + D*j for i in range(E)] for j in range(Ndir)]).T
	inputarray3 = inputarray1.copy() #need to copy since Tc constructor does not copy the memory

	a = Tc(3.,inputarray1)
	b = Tc(7.,inputarray2)

	a -= b

	print('inputarray3=\n',inputarray3)
	print('inputarray2=\n',inputarray2)
	print('inputarray3[:G] - inputarray2[:G]=\n',inputarray3[:G] - inputarray2[:G])
	print('a.tc[G:]=\n',a.tc[G:])
	print('-inputarray2[G:]=\n',-inputarray2[G:])
	assert a.t0 == -4.
	assert numpy.prod(a.tc[:G] == (inputarray3[:G] - inputarray2[:G]))
	assert numpy.prod(a.tc[G:] == -inputarray2[G:])


def test_incremental_multiplication_single_direction_Tc_Tc_same_order():
	inputarray1 = numpy.array([[0.,1.,2.]]).T
	inputarray2 = numpy.array([[7.,11.,13.]]).T
	inputarray3 = inputarray1.copy() #need to copy since Tc constructor does not copy the memory

	a = Tc(inputarray1)
	b = Tc(inputarray2)

	a *= b
	assert a.t0 == 0.
	assert a.tc[0] == ( inputarray3[0,0] * inputarray2[1,0] + inputarray3[1,0] * inputarray2[0,0] )
	assert a.tc[1] == ( inputarray3[0,0] * inputarray2[2,0]
	                  + inputarray3[1,0] * inputarray2[1,0]
					  + inputarray3[2,0] * inputarray2[0,0]  )

					  
def test_incremental_multiplication_single_direction_Tc_Tc_different_order():
	inputarray1 = numpy.array([[0.,1.,2.]]).T
	inputarray2 = numpy.array([[7.,11.]]).T
	inputarray3 = inputarray1.copy() #need to copy since Tc constructor does not copy the memory

	a = Tc(inputarray1)
	b = Tc(inputarray2)

	a *= b
	assert a.t0 == 0.
	assert a.tc[0] == ( inputarray3[0,0] * inputarray2[1,0] + inputarray3[1,0] * inputarray2[0,0] )
	assert a.tc[1] == ( inputarray3[1,0] * inputarray2[1,0]
					  + inputarray3[2,0] * inputarray2[0,0]  )

def test_incremental_multiplication_multiple_directions_Tc_Tc_same_order():
	inputarray1 = numpy.array([[0.,1.,2.],[3.,4.,5.],[2345.,12.,34.]])
	inputarray2 = numpy.array([[7.,11.,13.],[1.,2.,4.],[32.,4.,13.]])
	inputarray3 = inputarray1.copy() #need to copy since Tc constructor does not copy the memory

	a = Tc(3.,inputarray1)
	b = Tc(7.,inputarray2)

	a *= b

	print('inputarray3=\n',inputarray3)
	print('inputarray2=\n',inputarray2)
	print('3. * inputarray2[0,:] + inputarray3[0,:] * 7.=\n',3. * inputarray2[0,:] + inputarray3[0,:] * 7.)
	print('a.tc[0,:]=\n', a.tc[0,:])
	
	assert a.t0 == 21.
	assert numpy.prod(a.tc[0,:] == ( 3. * inputarray2[0,:] + inputarray3[0,:] * 7. ))
	assert numpy.prod(a.tc[1,:] == ( 3. * inputarray2[1,:]
	                  + inputarray3[0,:] * inputarray2[0,:]
					  + inputarray3[1,:] * 7.  )
					  )
	assert numpy.prod(a.tc[2,:] == (
	                    3. * inputarray2[2,:]
	                  + inputarray3[0,:] * inputarray2[1,:]
					  + inputarray3[1,:] * inputarray2[0,:]
					  + inputarray3[2,:] * 7.  )
					  )

def test_incremental_multiplication_multiple_directions_Tc_scalar():
	t0 = 2.
	tc = array([[1.,2,3],[4,5,6]])
	a = Tc(t0,tc.copy())
	a *= 2.
	print('a=',a)
	print('a.tc=',a.tc)
	print('2*tc=',2.*tc)
	assert a == 4.
	assert prod(a.tc == 2.*tc)


					  

def test_incremental_division_single_direction_Tc_Tc_same_order():
	inputarray1 = numpy.array([[1.,1.,2.]]).T
	inputarray2 = numpy.array([[7.,11.,13.]]).T
	inputarray3 = inputarray1.copy() #need to copy since Tc constructor does not copy the memory

	a = Tc(inputarray1)
	b = Tc(inputarray2)

	a /= b
	print('a.tc=\n',a.tc)
	print('a.tc true=\n', '[',( 1./inputarray2[0,0] *( inputarray3[1,0] - a.t0 * inputarray2[1,0] )),',',( 1./inputarray2[0,0] *( inputarray3[2,0] - a.t0 * inputarray2[2,0] - a.tc[0] * inputarray2[1,0] )), ']')
	assert a.t0 == inputarray3[0,0]/inputarray2[0,0]
	assert abs(a.tc[0] - ( 1./inputarray2[0,0] *( inputarray3[1,0] - a.t0 * inputarray2[1,0] )))<10**(-8)
	assert abs(a.tc[1] - ( 1./inputarray2[0,0] *( inputarray3[2,0] - a.t0 * inputarray2[2,0] - a.tc[0] * inputarray2[1,0] )))<10**(-8)

def test_incremental_division_single_direction_Tc_Tc_different_order():
	inputarray1 = numpy.array([[1.,1.,2.]]).T
	inputarray2 = numpy.array([[7.,11.]]).T
	inputarray3 = inputarray1.copy() #need to copy since Tc constructor does not copy the memory

	a = Tc(inputarray1)
	b = Tc(inputarray2)

	a /= b
	print('a.tc=\n',a.tc)
	print('a.tc true=\n', '[',( 1./inputarray2[0,0] *( inputarray3[1,0] - a.t0 * inputarray2[1,0] )),',',( 1./inputarray2[0,0] *( inputarray3[2,0] - a.tc[0] * inputarray2[1,0] )), ']')
	assert a.t0 == inputarray3[0,0]/inputarray2[0,0]
	assert abs(a.tc[0] - ( 1./inputarray2[0,0] *( inputarray3[1,0] - a.t0 * inputarray2[1,0] )))<10**(-8)
	assert abs(a.tc[1] - ( 1./inputarray2[0,0] *( inputarray3[2,0] -  a.tc[0] * inputarray2[1,0] )))<10**(-8)









# binary operators
def test_operators_single_direction_Tc_Tc_same_order():
	D = 4
	inputarray1 = numpy.array([1.* i+12 for i in range(D)])
	inputarray2 = numpy.array([1. +  i for i in range(D)])
	a = Tc(inputarray1)
	b = Tc(inputarray2)

	# functional test
	c = a+b
	c = a-b
	c = a*b
	c = a/b

	# identity test
	c = a-a
	assert c.t0 == 0
	assert numpy.prod(c.tc == 0)

	c = a/a
	assert c.t0 == 1
	assert numpy.prod(c.tc == 0)



def test_addition_single_direction_Tc_Tc_different_order():
	D = 4
	E = 7
	G = min(D,E) - 1 # first element is t0, thefefore subtract 1
	inputarray1 = numpy.array([[1.* i for i in range(D)]]).T
	inputarray2 = numpy.array([[1. +  i for i in range(E)]]).T

	a = Tc(inputarray1)
	b = Tc(inputarray2)

	c = a+b

	print('a.tc=',a.tc)
	print('b.tc=',b.tc)
	print('c.tc=',c.tc)
	print('a.tc[:G]=',a.tc[:G])
	print('b.tc[:G]=',b.tc[:G])

	assert c.t0 == a.t0 + b.t0
	assert numpy.prod(c.tc[:G] == (a.tc[:G] + b.tc[:G]))
	assert numpy.prod(c.tc[G:] == (b.tc[G:]))

def test_addition_multiple_directions_Tc_float():
	t0 = 2.
	tc = array([[1.,2,3],[23,43,51]])
	a = Tc(t0,tc)
	c = a+2
	d = 2+a

	print('c.t0=',c.t0)
	print('t0=',t0+2)
	print('c.tc=',c.tc)
	print('tc=',tc)
	assert c.t0 == 4.
	assert prod(c.tc == tc)

	assert d.t0 == 4.
	assert prod(d.tc == tc)

	

def test_division_single_direction_Tc_Tc_different_order():
	a = Tc(1,[[0.]])
	b = Tc(3.,[[5.],[7.]])

	c = a/b

	print(' c.tc[0,0]=', c.tc[0,0])
	print('-(a.t0/b.t0**2)*b.tc[0,0]=',-(a.t0/b.t0**2)*b.tc[0,0])
	print('c.tc[1,0]=',c.tc[1,0])
	print('( - (a.t0/b.t0**2)*b.tc[1,0] + 2*(a.t0/b.t0**3)*b.tc[1,0]**2 )=',( - (a.t0/b.t0**2)*b.tc[1,0] + (a.t0/b.t0**3)*b.tc[0,0]**2 ))
	
	assert c.t0 == a.t0 / b.t0
	assert abs(c.tc[0,0]  + (a.t0/b.t0**2)*b.tc[0,0]) < 10**-6
	assert abs(c.tc[1,0]  -( - (a.t0/b.t0**2)*b.tc[1,0] + (a.t0/b.t0**3)*b.tc[0,0]**2 )) < 10**-6

# unary operators

def test_sqrt():
	a = Tc(2.25)
	b = sqrt(a)
	print(a,b)
	assert sqrt(a.t0) == b.t0

	a = Tc([2.25,1.,0.])
	b = sqrt(a)
	print(a,b)

	print(0.5*a.t0**(-0.5))
	print(-0.25*a.t0**(-1.5)/2.)
	
	assert sqrt(a.t0) == b.t0
	assert 0.5*a.t0**(-0.5) == b.tc[0,0]
	assert abs(-0.25*a.t0**(-1.5) - 2*b.tc[1,0])<10**-3

def test_integer_power():
	a = Tc([2.25,1.,0.])
	b = a**3
	c = a*a*a
	assert b.t0 == c.t0
	assert prod(b.tc[:] == c.tc[:])


def test_exponential():
	a = Tc([2.25,1.,0.])
	b = exp(a)
	print(b)
	assert b.t0 == exp(a.t0)
	assert b.tc[0,0] == exp(a.t0)
	assert 2*b.tc[1,0] == exp(a.t0)

def test_logarithm():
	a = Tc([23.,1.,0.])
	b = log(a)

	print(b)
	assert b.t0 == log(a.t0)
	assert b.tc[0,0] == 1./a.t0
	assert 2*b.tc[1,0] == -1./(a.t0*a.t0)

def test_sin_and_cos():
	a = Tc([23.,1.,0.])
	s = sin(a)
	c = cos(a)

	assert s.t0 == sin(a.t0)
	assert c.t0 == cos(a.t0)

	assert s.tc[0,0] == cos(a.t0)
	assert c.tc[0,0] == -sin(a.t0)

	assert 2*s.tc[1,0] == -sin(a.t0)
	assert 2*c.tc[1,0] == -cos(a.t0)


# conditional operators
def test_lt_conditional():
	a = Tc(1,[[1,2,3]])
	b = Tc(1,[[1,2,3]])
	c = Tc(2,[[1,2,3]])
	d = Tc(-1,[[1,2,3]])

	# < operator
	assert not (a<b)
	assert not (b<a)
	assert not (a<d)
	assert     (a<c)

	# <= operator
	assert     (a<=b)
	assert     (b<=a)
	assert     (a<=c)
	assert not (a<=d)

	# == operator
	assert     (a==b)
	assert not (a==c)
	assert not (a==d)

	# != operator
	assert not (a!=b)
	assert     (a!=c)
	assert     (a!=d)

	# >= operator
	assert     (a>=b)
	assert     (b>=a)
	assert     (a>=d)
	assert not (a>=c)

	# > operator
	assert not (a>b)
	assert not (b>a)
	assert     (a>d)
	assert not (a>c)

#############################################################
#   TESTING CLASS Function AND CGraph
#############################################################


def test_graph_addition_with_constant():
	cg = CGraph()
	t0 = 2
	tc = array([[0,1],[1,0]])
	a = Function(Tc(t0,tc))
	b = a + 2
	c = 2 + a

	assert b.x.t0 == 4
	assert c.x.t0 == 4

	assert prod(b.x.tc == tc)
	assert prod(c.x.tc == tc)
	

def test_plotting_simple_cgraph():
	cg = CGraph()
	x = Function(Tc([11.,1.]))
	y = Function(Tc([13.,1.]))
	z = Function(Tc([13.,1.]))
	f = (x * y) + z*(x+y*(x*z))
	cg.independentFunctionList = [x,y]
	cg.dependentFunctionList = [z]
	cg.plot('trash/cg_example.png',method='circo')
	# no assert, this is only a functionality test,

def test_forward_mode():
	# first compute correct result by Taylor propagation
	x = Tc([11.,1.])
	y = Tc([13.,2.])
	z = Tc([13.,3.])
	f_tc = (x * y) + z*(x+y*(x*z))

	cg = CGraph()
	x = Function(Tc([11.]))
	y = Function(Tc([13.,1.]))
	z = Function(Tc([13.,1.,3,5,23]))
	f = (x * y) + z*(x+y*(x*z))
	cg.independentFunctionList = [x,y,z]
	cg.dependentFunctionList = [f]
	cg.forward([Tc([11.,1.]), Tc([13.,2.]), Tc([13.,3.])])

	print(f)
	print(f_tc)
	assert f.x.t0 == f_tc.t0
	assert f.x.tc[0] == f_tc.tc[0]

def test_reverse_mode_first_order():
	import sympy
	x,y,z = sympy.symbols('x','y','z')
	fs = (x * y) + z*(x+y*(x*z))
	gsx = fs.diff(x)
	gsy = fs.diff(y)
	gsz = fs.diff(z)
	dfdx = lambda x,y,z: eval(gsx.__str__())
	dfdy = lambda x,y,z: eval(gsy.__str__())
	dfdz = lambda x,y,z: eval(gsz.__str__())

	cg = CGraph()
	x = Function(Tc([11.]))
	y = Function(Tc([13.]))
	z = Function(Tc([17.]))
	f = (x * y) + z*(x+y*(x*z))
	cg.independentFunctionList = [x,y,z]
	cg.dependentFunctionList = [f]
	cg.reverse([Tc([1.])])
	#print f
	#print x
	#print y
	#print z

	print('x.xbar.t0=',x.xbar.t0)
	print('dfdx(11.,13.,17.)', dfdx(11.,13.,17.))
	assert x.xbar.t0 == dfdx(11.,13.,17.)
	assert y.xbar.t0 == dfdy(11.,13.,17.)
	assert z.xbar.t0 == dfdz(11.,13.,17.)


def test_reverse_mode_second_order():
	"""computing first column of the Hessian"""
	import sympy
	x = sympy.symbols('x')
	fs = x*x
	gsxx = fs.diff(x).diff(x)
	
	d2fdxdx = lambda x,y,z: eval(gsxx.__str__())

	cg = CGraph()
	x = Function(Tc([11.,1.]))
	f = x*x
	cg.independentFunctionList = [x]
	cg.dependentFunctionList = [f]
	cg.reverse([Tc(1.)])

	print(cg)


	print('x.xbar.tc[0,0]=',x.xbar.tc[0,0])
	print('d2fdxdx(11.,13.,17.)=',d2fdxdx(11.,13.,17.))

	assert x.xbar.tc[0,0] == d2fdxdx(11.,13.,17.)



def test_reverse_mode_second_order_two_variables():
	"""computing first column of the Hessian"""
	import sympy
	x,y = sympy.symbols('x','y')
	fs =  y*(y*x)
	gsxx = fs.diff(x).diff(x)
	gsxy = fs.diff(x).diff(y)
	
	d2fdxdx = lambda x,y: eval(gsxx.__str__())
	d2fdxdy = lambda x,y: eval(gsxy.__str__())


	cg = CGraph()
	x = Function(Tc([11.,1.]))
	y = Function(Tc([13.]))
	f = y*x*y
	cg.independentFunctionList = [x,y]
	cg.dependentFunctionList = [f]

	print(cg)
	cg.reverse([Tc(1.)])

	print(cg)


	print('x.xbar.tc[0,0]=',x.xbar.tc[0,0])
	print('d2fdxdx(11.,13.,17.)=',d2fdxdx(11.,13.))

	print('y.xbar.tc[0,0]=',y.xbar.tc[0,0])
	print('d2fdxdy(11.,13.,17.)',d2fdxdy(11.,13.))

	assert x.xbar.tc[0,0] == d2fdxdx(11.,13.)
	assert y.xbar.tc[0,0] == d2fdxdy(11.,13.)


def test_reverse_mode_second_order_three_variables():
	"""computing first column of the Hessian"""
	import sympy
	x,y,z = sympy.symbols('x','y','z')
	fs = (x * y) + z*(x+y*(x*z))
	gsxx = fs.diff(x).diff(x)
	gsxy = fs.diff(x).diff(y)
	gsxz = fs.diff(x).diff(z)
	
	d2fdxdx = lambda x,y,z: eval(gsxx.__str__())
	d2fdxdy = lambda x,y,z: eval(gsxy.__str__())
	d2fdxdz = lambda x,y,z: eval(gsxz.__str__())

	cg = CGraph()
	x = Function(Tc([11.,1.]))
	y = Function(Tc([13.]))
	z = Function(Tc([17.]))
	f = (x * y) + z*(x+y*(x*z))
	cg.independentFunctionList = [x,y,z]
	cg.dependentFunctionList = [f]

	print(cg)
	cg.reverse([Tc(1.)])
	print(cg)


	print('x.xbar.tc[0,0]=',x.xbar.tc[0,0])
	print('d2fdxdx(11.,13.,17.)=',d2fdxdx(11.,13.,17.))

	print('y.xbar.tc[0,0]=',y.xbar.tc[0,0])
	print('d2fdxdy(11.,13.,17.)',d2fdxdy(11.,13.,17.))

	print('z.xbar.tc[0,0]=',z.xbar.tc[0,0])
	print('d2fdxdz(11.,13.,17.)',d2fdxdz(11.,13.,17.))

	assert x.xbar.tc[0,0] == d2fdxdx(11.,13.,17.)
	assert y.xbar.tc[0,0] == d2fdxdy(11.,13.,17.)
	assert z.xbar.tc[0,0] == d2fdxdz(11.,13.,17.)

def test_inner_product_gradient():
	import numpy
	A = numpy.array([[11., 3.],[3.,17.]])
	def fun(x):
		return 0.5* numpy.dot(x, numpy.dot(A,x))
	cg = CGraph()
	x = numpy.array([Function(Tc(2.)), Function(Tc(7.))])
	f = fun(x)
	cg.independentFunctionList = x
	cg.dependentFunctionList = [f]

	cg.reverse([Tc(1.)])

	y = numpy.dot(A,[2.,7.])

	print('x[0].xbar=',x[0].xbar)
	print('x[1].xbar=',x[1].xbar)
	print('y[0]=',y[0])
	print('y[1]=',y[1])

	assert x[0].xbar.t0 == y[0]
	assert x[1].xbar.t0 == y[1]
	
def test_vector_forward_inner_product_hessian():
	import numpy
	A = numpy.array([[11., 3.],[3.,17.]])
	def fun(x):
		return 0.5* numpy.dot(x, numpy.dot(A,x))
	cg = CGraph()
	x = numpy.array([Function(Tc(2.,[[1.],[0.]])), Function(Tc(7.,[[0.],[1.]]))])
	f = fun(x)
	cg.independentFunctionList = x
	cg.dependentFunctionList = [f]

	cg.reverse([Tc(1.)])

	print('x[0].xbar.tc=',x[0].xbar.tc)
	print('x[1].xbar.tc=',x[1].xbar.tc)
	print('A=',A)

	assert numpy.prod(x[0].xbar.tc[:,0] == A[:,0])
	assert numpy.prod(x[1].xbar.tc[:,0] == A[:,1])

	cg.plot('trash/inner_product.png',method='circo')

def test_conditionals():
	def ge(a,b):
		if a>=b:
			return a*b
		else:
			return a/b

	def gt(a,b):
		if a>b:
			return a*b
		else:
			return a/b

	def le(a,b):
		if a<=b:
			return a*b
		else:
			return a/b

	def lt(a,b):
		if a<b:
			return a*b
		else:
			return a/b
		
	def eq(a,b):
		if a==b:
			return a*b
		else:
			return a/b

	def ne(a,b):
		if a!=b:
			return a*b
		else:
			return a/b
	
		
	cg = CGraph()
	a = Function(Tc([1.,2.,3.]))
	b = Function(Tc([34.,2.]))
	c = Function(Tc([34.,3.]))

	c = ge(a,b)
	d = ge(b,a)
	assert  c.x.t0 == 1./34
	assert  d.x.t0 == 34

	c = gt(a,b)
	d = gt(b,a)
	assert  c.x.t0 == 1./34
	assert  d.x.t0 == 34


	c = le(b,a)
	d = le(a,b)
	assert  c.x.t0 == 34
	assert  d.x.t0 == 34

	c = lt(b,a)
	d = lt(a,b)
	assert  c.x.t0 == 34
	assert  d.x.t0 == 34

	c = eq(a,b)
	d = eq(b,c)
	assert  c.x.t0 == 1./34
	assert  d.x.t0 == 34**2

	c = ne(a,b)
	d = ne(b,c)
	assert  c.x.t0 == 34
	assert  d.x.t0 == 1
	



def test_graph__sqrt():
	cg = CGraph()
	x = Function(Tc([121.,1.,0.]))
	f = sqrt(x)
	#f = exp(cos(sin(x)+y)+x)
	cg.independentFunctionList = [x]
	cg.dependentFunctionList = [f]
	cg.reverse([Tc(1)])

	print('x.x=\n',x.x)
	print('x.xbar=\n',x.xbar)
	print('x.bar.tc[0,0]=',x.xbar.tc[0,0])
	print('0.25 * x.x.t0**(-1.5)=',0.25 * x.x.t0**(-1.5))

	print('2* x.xbar.tc[1,0]=',2* x.xbar.tc[1,0])
	print(' 3./8 * x.x.t0**-2.5=', 3./8 * x.x.t0**-2.5)

	print(cg)
	
	assert x.xbar.t0 == 0.5 / sqrt(x.x.t0)
	assert abs(x.xbar.tc[0,0] + 0.25 * x.x.t0**(-1.5)) < 10**-6
	assert abs( 2* x.xbar.tc[1,0] - 3./8 * x.x.t0**-2.5) < 10**-6


def test_graph_sin():
	cg = CGraph()
	x = Function(Tc([1.,1.,0.]))
	f = sin(x)
	#f = exp(cos(sin(x)+y)+x)
	cg.independentFunctionList = [x]
	cg.dependentFunctionList = [f]
	cg.reverse([Tc(1.)])

	print('x.xbar.t0=',x.xbar.t0)
	print('cos(x.x.t0)=',cos(x.x.t0))

	print(' x.xbar.tc[0,0]=', x.xbar.tc[0,0])
	print('-sin(x.x.t0)=',-sin(x.x.t0))

	assert x.xbar.t0 == cos(x.x.t0)
	assert x.xbar.tc[0,0] == -sin(x.x.t0)

	assert 2.*x.xbar.tc[1,0] == -cos(x.x.t0)
	
	
def test_graph_exp():
	cg = CGraph()
	x = Function(Tc([1.,1.,0.]))
	f = exp(x)
	cg.independentFunctionList = [x]
	cg.dependentFunctionList = [f]
	cg.reverse([Tc(1.)])

	assert x.xbar.t0 == exp(x.x.t0)
	assert x.xbar.tc[0,0] == exp(x.x.t0)
	assert 2*x.xbar.tc[1,0] == exp(x.x.t0)


def test_graph_plotting_all_implemented_functions():
	A = array([[11., 3.],[3.,17.]])
	def fun(x):
		return 0.5* dot(x, dot(A,x))
	
	cg = CGraph()
	x = Function(Tc([1.,1.,0.]))
	y = Function(Tc([5.,1.,0.]))
	g = fun([x,y])
	f = sqrt(exp(cos(sin(x)/y)-x))
	f = f*g
	cg.independentFunctionList = [x]
	cg.dependentFunctionList = [f]
	cg.reverse([Tc(1)])
	cg.plot('trash/cgraph_all_implemented_functions.png',method='dot')
	cg.plot('trash/cgraph_all_implemented_functions.svg',method='dot')


#############################################################
#   TESTING HIGH LEVEL FUNCTIONS
#############################################################

def test_gradient_by_taping_then_gradient_from_graph():

	# defining the function
	A = array([[11., 3.],[3.,17.]])
	def fun(x):
		return 0.5* dot(x, dot(A,x))

	# tape the function
	x = array([1.,2.])
	cg = tape(fun,x)

	# compute gradient
	g = gradient_from_graph(cg)

	g_true = dot(A,x)
	assert prod(g == g_true)

def test_hessian():
	# defining the function
	A = array([[11., 3.],[3.,17.]])
	def fun(x):
		return 0.5* dot(x, dot(A,x))

	# compute the Hessian
	x = array([3.,7.])
	H = hessian(fun,x)
	print(H)

	assert prod(H == A)
	
