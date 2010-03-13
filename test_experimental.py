from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utp.utpm import *


class Test_Experimental(TestCase):
    
    def test_pb_cholesky(self):
        D,P,N = 3, 2, 10
        tmp = numpy.random.rand(*(D,P,N,N))
        A = UTPM(tmp)
        A = UTPM.dot(A.T,A)

        L = UTPM.cholesky(A)
        assert_array_almost_equal( A.data, UTPM.dot(L,L.T).data)
        
        

        


if __name__ == "__main__":
    run_module_suite() 
