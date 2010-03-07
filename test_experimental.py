from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utp.utpm import *


class Test_Eigenvalue_Decomposition_with_degenerate_Eigenvalues(TestCase):
    
    def test_first_order_two_blocks(self):
        D,P,N = 2,1,3
        X = UTPM(numpy.zeros((D,P,N,N)))
        
        # create orthogonormal transformation matrix
        T = UTPM.qr(UTPM(numpy.random.rand(*(D,P,N,N))))[0]
        
        # set diagonal elements of X
        X.data[0,0,0,0] = 1.
        X.data[0,0,1,1] = 1.
        X.data[0,0,2,2] = 2.
        X.data[1,0,0,0] = 3.
        X.data[1,0,1,1] = 5.
        X.data[1,0,2,2] = 7.
        
        
        # similarity transformation to get a matrix Y with the same eigenvalues
        # as X but not already diagonal
        # Y = T X T.T
        Y = UTPM.dot(T, UTPM.dot(X,T.T))
        
        tmp, Q_tilde = numpy.linalg.eigh(Y.data[0,0])
        
        # sanity checks for degree d = 0
        tmp = numpy.diag(tmp)
        assert_array_almost_equal(tmp, X.data[0,0])
        
        # compute first order coefficient
        dD = numpy.dot(Q_tilde.T, numpy.dot(Y.data[1,0], Q_tilde))
        
        U = numpy.linalg.eigh(dD)[1]
        
        print U
        
        Q = numpy.dot(Q_tilde, U.T)
        dD = numpy.dot(Q.T, numpy.dot(Y.data[1,0], Q))
        
        

        


if __name__ == "__main__":
    run_module_suite() 
