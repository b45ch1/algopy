from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utp.utpm import *


class Test_Experimental(TestCase):
    
    def test_choleksy_decomposition(self):
        D,P,N = 2, 1, 4
        tmp = numpy.random.rand(*(D,P,N,N))
        A = UTPM(tmp)
        A = UTPM.dot(A.T,A)
        
        L = UTPM.cholesky(A)
        
        print 'L=',L

        
        

        


if __name__ == "__main__":
    run_module_suite() 
