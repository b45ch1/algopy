from numpy.testing import *
import numpy

from algopy.utp.utpm import *


class TestMatPoly(TestCase):
    def test_UTPM_in_a_stupid_way(self):
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
        AZ = AX.T
        AX = AX.set_zero()
        
        
    def test_dot_output_shapes(self):
        D,P,N,M = 2,3,4,5
        X = 2 * numpy.random.rand(D,P,N,M)
        Y = 3 * numpy.random.rand(D,P,M,N)
        A = 3 * numpy.random.rand(D,P,M,N)
        x = 2 * numpy.random.rand(D,P,N)
        y = 2 * numpy.random.rand(D,P,N)

        aX = UTPM(X)
        aY = UTPM(Y)
        aA = UTPM(A)
        ax = UTPM(x)
        ay = UTPM(y)
        
        assert_array_equal( aX.dot(aY).tc.shape, (D,P,N,N))
        assert_array_equal( aY.dot(aX).tc.shape, (D,P,M,M))
        assert_array_equal( aA.dot(ax).tc.shape, (D,P,M))
        assert_array_equal( ax.dot(ay).tc.shape, (D,P,1))

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
        
    def test_getitem_single_element(self):
        X  = numpy.zeros((2,3,4,5))
        X2 = X.copy()
        X2[:,:,0,0] += 1
        AX = UTPM(X)
        AY = AX[0,0]
        AY.tc[:,:] += 1.
        assert_array_almost_equal(X,X2)
        
    def test_getitem_slice(self):
        X  = numpy.zeros((2,3,4,5))
        X2 = X.copy()
        X2[:,:,0:2,1:3] += 1
        AX = UTPM(X)
        AY = AX[0:2,1:3]
        AY.tc[:,:] += 1.
        assert_array_almost_equal(X,X2)        
        
    def test_setitem(self):
        D,P,N,M = 2,3,4,4
        X  = numpy.zeros((D,P,N,M))
        X2 = X.copy()
        for n in range(N):
            X2[:,:,n,n] = 1.
        Y  = numpy.ones((D,P))
        
        AX = UTPM(X)
        AY = UTPM(Y)

        for n in range(N):
            AX[n,n] = AY
        
        assert_array_almost_equal(X,X2)
        
    def test_reshape(self):
        D,P,N,M = 2,3,4,5
        X  = numpy.zeros((D,P,N,M))
        AX = UTPM(X)
        AY = AX.reshape((5,4))
        assert_array_equal(AY.tc.shape, (2,3,5,4))
        assert AY.tc.flags['OWNDATA']==False
        
        
        
    def test_transpose(self):
        D,P,N,M = 2,3,4,5
        X  = numpy.zeros((D,P,N,M))
        AX = UTPM(X)
        assert_array_equal(AX.T.tc.shape, (D,P,M,N))
        
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
        (D,P,N,M) = 3,3,30,1
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
