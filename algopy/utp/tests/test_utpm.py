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
        D,P,N,M = 2,3,4,5
        X = 2 * numpy.random.rand(D,P,N,M)

        AX = UTPM(X)
        AY1 = 2 + AX
        AY2 = 2 - AX
        AY3 = 2 * AX
        AY4 = 2 / AX
        AY5 = AX + 2
        AY6 = AX - 2
        AY7 = AX * 2
        AY8 = AX / 2
        
        AX1 = UTPM(X.copy())
        AX2 = UTPM(X.copy())
        AX3 = UTPM(X.copy())
        AX4 = UTPM(X.copy())
        
        AX1 += 2
        AX2 -= 2
        AX3 *= 2
        AX4 /= 2
        
        Z1 = X.copy()
        Z2 = - X.copy()
        Z3 = X.copy()
        Z4 = 1./X.copy()
        Z5 = X.copy()
        Z6 = X.copy()
        Z7 = X.copy()
        Z8 = X.copy()
            
        for d in range(D):    
            for p in range(P):
                if d == 0:
                    Z1[0,p,...] += 2
                    Z2[0,p,...] += 2
                    Z5[0,p,...] += 2
                    Z6[0,p,...] -= 2
                Z3[d,p,...] *= 2
                Z7[d,p,...] *= 2
                Z8[d,p,...] /= 2
                
        assert_array_almost_equal(AY1.tc, Z1 )
        assert_array_almost_equal(AY2.tc, Z2 )
        assert_array_almost_equal(AY3.tc, Z3 )
        # assert_array_almost_equal(AY4.tc, Z4 )
        assert_array_almost_equal(AY5.tc, Z5 )
        assert_array_almost_equal(AY6.tc, Z6 )
        assert_array_almost_equal(AY7.tc, Z7 )
        assert_array_almost_equal(AY8.tc, Z8 )
        
        assert_array_almost_equal(AX1.tc, AY5.tc )
        assert_array_almost_equal(AX2.tc, AY6.tc )
        assert_array_almost_equal(AX3.tc, AY7.tc )
        assert_array_almost_equal(AX4.tc, AY8.tc )

        
    def test_array_operations(self):
        D,P,N,M = 2,3,4,5
        
        X = 2 * numpy.random.rand(D,P,N,M)
        Y = 3 * numpy.random.rand(N,M)
        AX = UTPM(X)
        AY1 = Y + AX
        AY2 = Y - AX
        AY3 = Y * AX
        AY4 = Y / AX
        AY5 = AX + Y
        AY6 = AX - Y
        AY7 = AX * Y
        AY8 = AX / Y
        
        AX1 = UTPM(X.copy())
        AX2 = UTPM(X.copy())
        AX3 = UTPM(X.copy())
        AX4 = UTPM(X.copy())
        
        AX1 += Y
        AX2 -= Y
        AX3 *= Y
        AX4 /= Y
        
        assert_array_almost_equal(AX1.tc, AY5.tc )
        assert_array_almost_equal(AX2.tc, AY6.tc )
        assert_array_almost_equal(AX3.tc, AY7.tc )
        assert_array_almost_equal(AX4.tc, AY8.tc )
        

        
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
        
    def test_setitem_iadd_scalar(self):
        D,P,N,M = 2,3,4,4
        X  = numpy.zeros((D,P,N,M))
        X2 = X.copy()
        for n in range(N):
            X2[0,:,n,n] += 2.
        
        AX = UTPM(X)
        for n in range(N):
            AX[n,n] += 2.
        
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
