from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utp.utpm import *
from algopy.globalfuncs import *

D,P,N = 4,1,100

A = UTPM(numpy.random.rand(D,P,N,N))
Q,R = qr(A)
l = UTPM(numpy.random.rand(D,P,N))
L = diag(l)

A = dot(Q, dot(L,Q.T))

l2,Q2 = eigh(A)

print l.data[:,:,numpy.argsort(l.data[0,0])] - l2.data



if __name__ == "__main__":
    run_module_suite() 


    
    
    
    
