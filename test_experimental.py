from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utp.utpm import *


class Test_Experimental(TestCase):
    
    def test_pb_cholesky(self):
        D,P,N = 1, 1, 3
        tmp = numpy.random.rand(*(D,P,N,N))
        A = UTPM(tmp)
        A = UTPM.dot(A.T,A)

        L = UTPM.cholesky(A)
        Lbar = UTPM(numpy.random.rand(*(D,P,N,N)))
        Abar = UTPM.pb_cholesky(Lbar, A, L)
        # print L
        
        

        


if __name__ == "__main__":
    run_module_suite() 
