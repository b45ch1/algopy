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


def test_trace():
	N1 = 2
	N2 = 3
	N3 = 4
	N4 = 5
	x = asarray(range(N1*N2*N3*N4))
	x = x.reshape((N1,N2,N3,N4))
	AX = Mtc(x)
	AY = AX.T
	AY.TC[0,0,2,0] = 1234
	assert AX.TC[0,0,0,2] == AY.TC[0,0,2,0]



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
	#assert False

def test_forward_UTPM_inv():
	X = numpy.zeros((3,1,2,2))
	X[0,0,:,:] = array([[10.,2.],[2.,11.]])
	X[1,0,:,:] = array([[1.,2.],[3.,4.]])
	X[2,0,:,:] = array([[5.,6.],[7.,8.]])
	cg = CGraph()
	FX = Function(Mtc(X))
	FZ = FX.inv()
	cg.independentFunctionList = [FX]
	cg.dependentFunctionList = [FZ]

	Z = numpy.zeros((3,1,2,2))
	Z[0,0,:,:] = numpy.linalg.inv(X[0,0,:,:])
	Z[1,0,:,:] = dot(dot(-Z[0,0,:,:],X[1,0,:,:]),Z[0,0,:,:])
	Z[2,0,:,:] = dot(-Z[0,0,:,:], dot(X[2,0,:,:],Z[0,0,:,:]) + dot(X[1,0,:,:], Z[1,0,:,:])  )

	assert sum( abs(Z[0,0,:,:] - FZ.x.TC[0,0,:,:])) < 10**-10
	assert sum( abs(Z[1,0,:,:] - FZ.x.TC[1,0,:,:])) < 10**-10
	assert sum( abs(Z[2,0,:,:] - FZ.x.TC[2,0,:,:])) < 10**-10


	



