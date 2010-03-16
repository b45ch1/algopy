from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utp.utpm import *


class Test_Experimental(TestCase):
    
    # def test_pb_cholesky(self):
        # D,P,N = 2, 1, 3
        # tmp = numpy.random.rand(*(D,P,N,N))
        # A = UTPM(tmp)
        # A = UTPM.dot(A.T,A)

        # L = UTPM.cholesky(A)
        # Lbar = UTPM(numpy.random.rand(*(D,P,N,N)))
        # for r in range(N):
            # for c in range(N):
                # Lbar[r,c] *= (r>=c)
        
        # # print Lbar
        # # print L
        
        # Abar = UTPM.pb_cholesky(Lbar, A, L)
        # print Abar
        
        # for p in range(P):
            # Ab = Abar.data[0,p]
            # Ad = A.data[1,p]

            # Lb = Lbar.data[0,p]
            # Ld = L.data[1,p]
            # assert_almost_equal(numpy.trace(numpy.dot(Ab.T,Ad)), numpy.trace(numpy.dot(Lb.T,Ld) ))






if __name__ == "__main__":
    run_module_suite() 
