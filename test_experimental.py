from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utp.utpm import *


class Test_Experimental(TestCase):
    
    # def test_pb_cholesky(self):
    #     D,P,N = 2, 1, 3
    #     tmp = numpy.random.rand(*(D,P,N,N))
    #     A = UTPM(tmp)
    #     A = UTPM.dot(A.T,A)

    #     L = UTPM.cholesky(A)
    #     Lbar = UTPM(numpy.random.rand(*(D,P,N,N)))
    #     for r in range(N):
    #         for c in range(N):
    #             Lbar[r,c] *= (r>=c)
        
    #     # print Lbar
    #     # print L
        
    #     Abar = UTPM.pb_cholesky(Lbar, A, L)
    #     print Abar
        
    #     for p in range(P):
    #         Ab = Abar.data[0,p]
    #         Ad = A.data[1,p]

    #         Lb = Lbar.data[0,p]
    #         Ld = L.data[1,p]
    #         assert_almost_equal(numpy.trace(numpy.dot(Ab.T,Ad)), numpy.trace(numpy.dot(Lb.T,Ld) ))


    def test_pullback_more_cols_than_rows(self):
        """
        A.shape = (3,11)
        """
        (D,P,M,N) = 2,1,3,11
        A_data = numpy.random.rand(D,P,M,N)
        
        # make A_data sufficiently regular
        for p in range(P):
            for m in range(M):
                A_data[0,p,m,m] += (N + 1)
        

        A = UTPM(A_data)

        Q,R = UTPM.qr(A)
        
        Qbar = UTPM(numpy.random.rand(D,P,M,M))
        Rbar = UTPM(numpy.random.rand(D,P,M,N))
        Rbar[:,M:] *= 0
        Abar = UTPM.pb_qr(Qbar, Rbar, A, Q, R)
        
        for p in range(P):
            Ab = Abar.data[0,p]
            Ad = A.data[1,p]
            Qb = Qbar.data[0,p]
            Qd = Q.data[1,p]
            Rb = Rbar.data[0,p]
            Rd = R.data[1,p]
            assert_almost_equal(numpy.trace(numpy.dot(Ab.T,Ad)), numpy.trace(numpy.dot(Qb.T,Qd)) + numpy.trace( numpy.dot(Rb.T,Rd)))
        
        


if __name__ == "__main__":
    run_module_suite() 
