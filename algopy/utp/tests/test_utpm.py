from numpy.testing import *
import numpy

from algopy.utp.utpm import *


class TestMatPoly(TestCase):
    def test_UTPM(self):
        """
        this checks _only_ if calling the operations is ok
        """
        X = 2 * numpy.random.rand(2,2,2,2)
        Y = 3 * numpy.random.rand(2,2,2,2)

        AX = UTPM(X)
        AY = UTPM(Y)
        AZ = AX + AY
        AZ = AX - AY
        AZ = AX * AY
        AZ = AX / AY
        AZ = AX.dot(AY)
        AZ = AX.inv()
        AZ = AX.trace()
        AZ = AX[0,0]
        AZ = AX.T
        AX = AX.set_zero()

    def test_scalar_operations(self):
        X = 2 * numpy.random.rand(2,2,2,2)

        AX = UTPM(X)
        AY = 2 + AX
        AY = 2 - AX
        AY = 2 * AX
        AY = 2 / AX
        AY = AX + 2
        AY = AX - 2
        AY = AX * 2
        AY = AX / 2
        
    def test_array_operations(self):
        
        X = 2 * numpy.random.rand(2,2,2,2)
        Y = 3 * numpy.random.rand(2,2)
        AX = UTPM(X)
        AY = Y + AX
        AY = Y - AX
        AY = Y * AX
        AY = Y / AX
        AY = AX + Y
        AY = AX - Y
        AY = AX * Y
        AY = AX / Y        
        

    def test_trace(self):
        N1 = 2
        N2 = 3
        N3 = 4
        N4 = 5
        x = numpy.asarray(range(N1*N2*N3*N4))
        x = x.reshape((N1,N2,N3,N4))
        AX = UTPM(x)
        AY = AX.T
        AY.tc[0,0,2,0] = 1234
        assert AX.tc[0,0,0,2] == AY.tc[0,0,2,0]
        
    def test_inv(self):
        (D,P,N,M) = 2,3,5,1
        A = UTPM(numpy.random.rand(D,P,N,N))
        Ainv = A.inv()
        
        Id = numpy.zeros((D,P,N,N))
        Id[0,:,:,:] = numpy.eye(N)
        assert_array_almost_equal(A.dot(Ainv).tc, Id)
        
    def test_solve(self):
        (D,P,N,M) = 4,3,30,1
        x = UTPM(numpy.random.rand(D,P,N,M))
        A = UTPM(numpy.random.rand(D,P,N,N))
        y = x.solve(A)
        x2 = A.dot(y)
        assert_array_almost_equal(x.tc, x2.tc, decimal = 4)
            


class TestCombineBlocks(TestCase):
    def test_convert(self):
        X1 = 2 * numpy.random.rand(2,2,2,2)
        X2 = 2 * numpy.random.rand(2,2,2,2)
        X3 = 2 * numpy.random.rand(2,2,2,2)
        X4 = 2 * numpy.random.rand(2,2,2,2)
        AX1 = UTPM(X1)
        AX2 = UTPM(X2)
        AX3 = UTPM(X3)
        AX4 = UTPM(X4)
        AY = combine_blocks([[AX1,AX2],[AX3,AX4]])

        assert_array_equal(numpy.shape(AY.tc),(2,2,4,4))


if __name__ == "__main__":
    run_module_suite()
