#!/usr/bin/env python
import numpy
import numpy.linalg
from numpy import *
import numpy.random


try:
	import sys
	sys.path = ['..'] + sys.path
	from algopy import *
except:
	from algopy import *


def test_tapeless_forward_UTPM():
	X = 2 * numpy.random.rand(2,2,2,2)
	Y = 3 * numpy.random.rand(2,2,2,2)

	AX = Mtc(X)
	AY = Mtc(Y)
	AZ = AX + AY
	AZ = AX - AY
	AZ = AX * AY
	AZ = AX / AY
	AZ = AX.dot(AY)
	AZ = AX.inv()
	AZ = AX.trace()
	AZ = AX[0,0]
	AZ = AX.T
	AX = AX.set_zero()
	print 'AX=',AX
	#print 'AY=',AY
	print 'AZ=',AZ




def test_forward_UTPM_add():
	X = 2 * numpy.random.rand(2,2,2,2)
	Y = 3 * numpy.random.rand(2,2,2,2)
	
	cg = CGraph()
	FX = Function(Mtc(X))
	FY = Function(Mtc(X))
	FZ = FX.dot(FY)
	cg.independentFunctionList = [FX,FY]
	cg.dependentFunctionList = [FZ]

	print FZ
	assert False



