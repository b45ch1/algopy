#!/usr/bin/env python
import numpy
import numpy.linalg
from numpy import *
import numpy.random
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal, assert_equal

try:
	import sys
	sys.path = ['..'] + sys.path
	from algopy import *
except:
	from algopy import *



# TESTING HELPER FUNCTIONS
# ------------------------
def test_convert_on_MTC():
	D = [3,3,3]
	P = [4,4,4]
	N = [2,3,5]
	M = [1,1,1]

	Ms = []
	for i in range(3):
		Ms.append([Mtc(numpy.random.rand(D[i],P[i],N[i],M[i]))])

	MX = convert(Ms)
	
	# check the left-top block
	assert_array_almost_equal(MX.TC[:,:,:N[0],:M[0]], Ms[0][0].TC[:,:,:,:])

def test_convert_on_Function():
	D = [3,3,3]
	P = [4,4,4]
	N = [2,3,5]
	M = [1,1,1]

	cg = CGraph()

	Ms = []
	for i in range(3):
		Ms.append([Function(Mtc(numpy.random.rand(D[i],P[i],N[i],M[i])))])

	MX = convert(Ms)
	
	# check the left-top block
	assert_array_almost_equal(MX.TC[:,:,:N[0],:M[0]], Ms[0][0].x.TC[:,:,:,:])
	

# TESTING MATRIX POLYNOMIAL COMPUTATIONS
# --------------------------------------

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
	#print 'AX=',AX
	#print 'AY=',AY
	#print 'AZ=',AZ


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




# TESTING MATRIX AD
# -----------------

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



def test_plot_computational_graph():
	X = 2 * numpy.random.rand(2,2,2,2)
	Y = 2 * numpy.random.rand(2,2,2,2)

	AX = Mtc(X)
	AY = Mtc(Y)
	cg = CGraph()
	FX = Function(AX)
	FY = Function(AY)

	FX = FX*FY
	FX = FX.dot(FY) + FX.transpose()
	FX = FY + FX * FY
	FY = FX.inv()
	FY = FY.transpose()
	FZ = FX * FY

	FW = Function([[FX, FZ], [FZ, FY]])

	FTR = FW.trace()
	
	cg.independentFunctionList = [FX, FY]
	cg.dependentFunctionList = [FTR]
	
	cg.plot(filename = 'trash/computational_graph_circo.png', method = 'circo' )
	cg.plot(filename = 'trash/computational_graph_circo.svg', method = 'circo' )
	cg.plot(filename = 'trash/computational_graph_dot.png', method = 'dot' , orientation = 'LR')
	cg.plot(filename = 'trash/computational_graph_dot.svg', method = 'dot' )

def test_forward_reverse_combine():
	D = [1,1,1]
	P = [1,1,1]
	N = [1,2,3]
	M = [1,1,1]

	cg = CGraph()

	Fs = []
	for i in range(3):
		Fs.append([Function(Mtc(numpy.random.rand(D[i],P[i],N[i],M[i])))])

	FX = Function(Fs)
	FXT = FX.T
	FTR = FXT.dot(FX).trace()
	cg.independentFunctionList = [Fs[i][0] for i in range(3)]
	cg.dependentFunctionList   = [FTR]

	cg.plot(filename = 'trash/test_forward_reverse_combine.png', method = 'circo' )

	cg.reverse([Mtc(numpy.array([[[[1.]]]]))])

	print cg

	#print cg.independentFunctionList[0]
	assert False
	# check the left-top block
	#assert_array_almost_equal(MX.TC[:,:,:N[0],:M[0]], Ms[0][0].x.TC[:,:,:,:])

	
