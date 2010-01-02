from adolc import adouble
from adolc.cgraph import AdolcProgram
from adolc.linalg import qr
import numpy
import numpy.testing
from algopy.utp.utpm import UTPM
from time import time


D,P,N = 3,1,50

#----------------------------------------------
# STEP 1:
# QR decomposition by Givens Rotations
# using pyadolc for the differentiation
#----------------------------------------------

# trace QR decomposition with adolc
A = numpy.random.rand(N,N)
AP = AdolcProgram()
AP.trace_on(1)
aA = adouble(A)
AP.independent(aA)
aQ, aR = qr(aA)
AP.dependent(aQ)
AP.dependent(aR)
AP.trace_off()

# compute push forward
VA = numpy.random.rand(N,N,P,D)

tic = time()
out = AP.forward([A],[VA])
toc = time()

runtime_pyadolc = toc - tic

#----------------------------------------------
# STEP 2:
# QR decomposition using LAPACK
# using algopy for the differentiation
#----------------------------------------------

A = UTPM(numpy.random.rand(D,P,N,N))
tic = time()
Q,R = UTPM.qr(A)
toc = time()

runtime_algopy = toc - tic

print runtime_algopy/runtime_pyadolc





