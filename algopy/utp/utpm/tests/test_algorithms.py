from numpy.testing import *
import numpy

from algopy.utp.utpm import *
from algopy.utp.utpm.algorithms import *

class Test_Helper_Functions(TestCase):
    def test_vdot(self):
        (D,P,N,M) = 4,3,2,5
        A = numpy.array([ i for i in range(D*P*N*M)],dtype=float)
        A = A.reshape((D,P,N,M))
        B = A.transpose((0,1,3,2)).copy()

        R  = vdot(A[0],B[0])
        R2 = numpy.zeros((P,N,N))
        for p in range(P):
            R2[p,:,:] = numpy.dot(A[0,p],B[0,p])

        S  = vdot(A,B)
        S2 = numpy.zeros((D,P,N,N))
        for d in range(D):
            for p in range(P):
                S2[d,p,:,:] = numpy.dot(A[d,p],B[d,p])

        assert_array_almost_equal(R,R2)
        assert_array_almost_equal(S,S2)      
       
    def test_triple_truncated_dot(self):
        D,P,N,M = 3,1,1,1
        A = numpy.random.rand(D,P,N,M)
        B = numpy.random.rand(D,P,N,M)
        C = numpy.random.rand(D,P,N,M)

        S = A[0]*B[1]*C[1] + A[1]*B[0]*C[1] + A[1]*B[1]*C[0]
        R = truncated_triple_dot(A,B,C,2)

        assert_array_almost_equal(R,S)

        D,P,N,M = 4,1,1,1
        A = numpy.random.rand(D,P,N,M)
        B = numpy.random.rand(D,P,N,M)
        C = numpy.random.rand(D,P,N,M)

        S = A[0]*B[1]*C[2] + A[0]*B[2]*C[1] + \
            A[1]*B[0]*C[2] + A[1]*B[1]*C[1] + A[1]*B[2]*C[0] +\
            A[2]*B[1]*C[0] + A[2]*B[0]*C[1]
        R = truncated_triple_dot(A,B,C, 3)

        assert_array_almost_equal(R,S)        

class Test_push_forward_class_functions(TestCase):
    """
    Test the push forward class functions that operate directly on data.
    """

    def test__idiv(self):
        X_data = 2 * numpy.random.rand(2,2,2,2)
        Z_data = 3 * numpy.random.rand(2,2,2,2)
        Z2_data = Z_data.copy()

        UTPM._idiv(Z_data, X_data)

        X = UTPM(X_data)
        Z = UTPM(Z2_data)

        Z/=X

        assert_array_almost_equal(Z_data, Z.data)

    def test__div(self):
        X_data = 2 * numpy.random.rand(2,2,2,2)
        Y_data = 3 * numpy.random.rand(2,2,2,2)
        Z_data = numpy.zeros((2,2,2,2))

        X = UTPM(X_data)
        Y = UTPM(Y_data)

        Z = X/Y

        UTPM._div(X_data, Y_data, out = Z_data)

        assert_array_almost_equal(Z_data, Z.data)

    def test__transpose(self):
        D,P,M,N = 2,3,4,5
        X_data = numpy.random.rand(D,P,N,M)
        Y_data = UTPM._transpose(X_data)

        assert_array_almost_equal(X_data.transpose((0,1,3,2)), Y_data) 





if __name__ == "__main__":
    run_module_suite()