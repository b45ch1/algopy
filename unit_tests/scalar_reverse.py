#!/usr/bin/env python
import sys
import numpy
import numpy.linalg

#sys.path = ['..'] + sys.path
from reverse_mode import *


# TESTING ALL FUNCTIONS FOR BASIC FUNCTIONALITY

# testing the __init_function
def test_constructor_single_direction_list_as_input():
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
	return
	a = Tc(3.,1.,2.)
	assert a.t0 == 3.
	assert numpy.prod(a.tc[:] == numpy.array([[1.,2.]]).T)


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

def test_incremental_addition_multiple_directions_Tc_Tc_same_order():
	D = 4
	Ndir = 3
	inputarray1 = numpy.array([[1.* i + D*j for i in range(D)] for j in range(Ndir) ]).T
	inputarray2 = numpy.array([[1. +  i + D*j for i in range(D)] for j in range(Ndir)]).T
	inputarray3 = inputarray1.copy() #need to copy since Tc constructor does not copy the memory

	a = Tc(3.,inputarray1)
	b = Tc(7.,inputarray2)

	a += b

	print 'inputarray3=\n',inputarray3
	print 'inputarray2=\n',inputarray2
	print  'inputarray3[1:] + inputarray2[1:]=\n',inputarray3[:] + inputarray2[:]
	assert a.t0 == 10.
	assert numpy.prod(a.tc == (inputarray3[:] + inputarray2[:]))


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

def test_incremental_multiplication_multiple_directions_Tc_Tc_same_order():
	inputarray1 = numpy.array([[0.,1.,2.],[3.,4.,5.],[2345.,12.,34.]])
	inputarray2 = numpy.array([[7.,11.,13.],[1.,2.,4.],[32.,4.,13.]])
	inputarray3 = inputarray1.copy() #need to copy since Tc constructor does not copy the memory

	a = Tc(3.,inputarray1)
	b = Tc(7.,inputarray2)

	a *= b

	print 'inputarray3=\n',inputarray3
	print 'inputarray2=\n',inputarray2
	print '3. * inputarray2[0,:] + inputarray3[0,:] * 7.=\n',3. * inputarray2[0,:] + inputarray3[0,:] * 7.
	print 'a.tc[0,:]=\n', a.tc[0,:]
	
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
					  


def test_incremental_division_single_direction_Tc_Tc_same_order():
	inputarray1 = numpy.array([[1.,1.,2.]]).T
	inputarray2 = numpy.array([[7.,11.,13.]]).T
	inputarray3 = inputarray1.copy() #need to copy since Tc constructor does not copy the memory

	a = Tc(inputarray1)
	b = Tc(inputarray2)

	a /= b

	assert a.t0 == inputarray3[0,0]/inputarray2[0,0]
	assert a.tc[0] == ( 1./inputarray2[0,0] *( inputarray3[1,0] - a.t0 * inputarray2[1,0] ))
	assert a.tc[1] == ( 1./inputarray2[0,0] *( inputarray3[2,0] - a.t0 * inputarray2[2,0] - a.tc[0] * inputarray2[1,0] ))

# binary operators
def test_operators_single_direction_Tc_Tc_same_order():
	D = 4
	inputarray1 = numpy.array([1.* i for i in range(D)])
	inputarray2 = numpy.array([1. +  i for i in range(D)])
	a = Tc(inputarray1)
	b = Tc(inputarray2)
	c = a+b
	c = a-b
	c = a*b
	print 'a=',a
	print 'b=',b
	print 'c=',c
	# no assert, this is only a functionality test,
	# correctness is tested in the incremental implementation

def test_addition_single_direction_Tc_Tc_different_order():
	D = 4
	E = 7
	G = min(D,E) - 1 # first element is t0, thefefore subtract 1
	inputarray1 = numpy.array([[1.* i for i in range(D)]]).T
	inputarray2 = numpy.array([[1. +  i for i in range(E)]]).T

	a = Tc(inputarray1)
	b = Tc(inputarray2)

	c = a+b

	print 'a.tc=',a.tc
	print 'b.tc=',b.tc
	print 'c.tc=',c.tc
	print 'a.tc[:G]=',a.tc[:G]
	print 'b.tc[:G]=',b.tc[:G]

	assert c.t0 == a.t0 + b.t0
	assert numpy.prod(c.tc[:G] == (a.tc[:G] + b.tc[:G]))
	assert numpy.prod(c.tc[G:] == (b.tc[G:]))
