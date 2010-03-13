from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utp.utpm import *


class Test_Experimental(TestCase):
    
    # def test_pb_cholesky(self):
    #     D,P,N = 1, 1, 3
    #     tmp = numpy.random.rand(*(D,P,N,N))
    #     A = UTPM(tmp)
    #     A = UTPM.dot(A.T,A)

    #     L = UTPM.cholesky(A)
    #     Lbar = UTPM(numpy.random.rand(*(D,P,N,N)))
    #     Abar = UTPM.pb_cholesky(Lbar, A, L)
    #     print Abar
        
        
    def test_pb_qr_alternative(self):
        (D,P,M,N) = 2,1,3,3

        A_data = numpy.random.rand(D,P,M,N)

        # make A_data sufficiently regular
        for p in range(P):
            for n in range(N):
                A_data[0,p,n,n] += (N + 1)

        A = UTPM(A_data)

        # STEP 1: push forward
        Q,R = UTPM.qr(A)

        # STEP 2: pullback

        Qbar_data = numpy.random.rand(*Q.data.shape)
        Rbar_data = numpy.random.rand(*R.data.shape)

        for r in range(N):
            for c in range(N):
                Rbar_data[:,:,r,c] *= (c>=r)

        Qbar = UTPM(Qbar_data)
        Rbar = UTPM(Rbar_data)

        Abar = UTPM.pb_qr(Qbar, Rbar, A, Q, R)
        
        print Abar
        # print Rbar
        print UTPM.dot(Q, Rbar)


if __name__ == "__main__":
    run_module_suite() 
