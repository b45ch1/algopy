from numpy.testing import *
import numpy

from algopy.utp.utpm import *


class TestFunctionOfJacobian(TestCase):
    def test_FtoJT(self):
        (D,P,N) = 2,5,5
        x = UTPM(numpy.random.rand(D,P,N))
        z = x.tc[1:,...].reshape((D-1,1,P,N))
        y = x.FtoJT()
        assert_array_equal(y.tc.shape, [1,1,5,5])
        assert_array_almost_equal(y.tc, z)
        
    def test_JTtoF(self):
        (D,P,N) = 2,5,5
        x = UTPM(numpy.random.rand(D,P,N))
        y = x.FtoJT()
        z = y.JTtoF()
        
        assert_array_equal(x.tc.shape, z.tc.shape)
       
        assert_array_almost_equal(x.tc[1:,...], z.tc[:-1,...])
        
    def test_shifted_multiplication_from_left(self):
        D,P,N = 3,1,1
        x = numpy.zeros((D,P,N))
        x[:,0,0] = [1,11,13]
        xbar = numpy.zeros((D-1,P,N))
        xbar[:,0,0] = [5,7]

        x = UTPM(x)
        xbar = UTPM(xbar,shift=1)
        
        zbar = xbar * x
        assert_array_almost_equal(zbar.tc[:,0,0], [55,142])
        
        
    def test_first_order_J(self):
        """
        function:        
            f = x * y
        gradient:
            g = [ y, x]
            
        analytical pull back:
            gbar.T d g = gbar1 dy  +  gbar2 dx
            
        numerical:
            gbar^T d d/dt f = gbar d/dt (1 d f)
        
        """    
        
        
        D,P = 2,2
        x = UTPM(numpy.zeros((D,P)))
        y = UTPM(numpy.zeros((D,P)))
        
        x0,y0 = 3,5
        
        x.tc[0,:] = x0
        x.tc[1,0] = 1
        y.tc[0,:] = y0
        y.tc[1,1] = 1        
        
        # forward
        f = x*y
        
        # reverse
        gbar = UTPM(numpy.random.rand(D-1,1,P), shift=1)
        fbar = gbar
        fbar.tc = fbar.tc.reshape((D-1,P))
        
        xbar1 = numpy.dot(fbar.tc[0,:],y.tc[1,:])
        ybar1 = numpy.dot(fbar.tc[0,:], x.tc[1,:])
        
        xbar3 = (fbar * y).tc[0,:].sum()
        ybar3 = (fbar * x).tc[0,:].sum()

        # analytical solution
        xbar2 = fbar.tc[0,1]
        ybar2 = fbar.tc[0,0]
        
        assert_almost_equal(xbar1,xbar2)
        assert_almost_equal(ybar1,ybar2)
        assert_almost_equal(ybar1,ybar3)
        assert_almost_equal(ybar1,ybar3)
        
    
    def test_first_order_J2(self):
        """
        function:        
            f = x**2 * y
        gradient:
            g = [2 x y, x**2]
            
        analytical pull back:
            gbar.T d g = (gbar1 2 x) dy + ( gbar1 2 y + 2 gbar2 x) dx 
        
        """
        
        D,P = 2,2
        x = UTPM(numpy.zeros((D,P)))
        y = UTPM(numpy.zeros((D,P)))
        
        x0,y0 = 3,5
        
        x.tc[0,:] = x0
        x.tc[1,0] = 1
        y.tc[0,:] = y0
        y.tc[1,1] = 1        
        
        # forward
        f = x*y*x
        
        # reverse of g = d/dt f
        gbar = UTPM(numpy.random.rand(D-1,1,P), shift=1)
        fbar = gbar
        fbar.tc = fbar.tc.reshape((D-1,P))
        
        # reverse of f
        xbar = 2 * x * y
        ybar = x * x
        
        # compute d d/dt f
        xbar1 = (fbar * xbar).tc[0,:].sum()
        ybar1 = (fbar * ybar).tc[0,:].sum()
        
        # analytical solution
        xbar2 = fbar.tc[0,0] * 2 * y0 + fbar.tc[0,1] * 2 * x0
        ybar2 = fbar.tc[0,0] * 2 * x0

        assert_almost_equal(xbar1,xbar2)
        assert_almost_equal(ybar1,ybar2)
        

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
        
        
    def test_generic_calling(self):
        """
        this checks _only_ if calling the operations is ok
        """
        X = 2 * numpy.random.rand(2,2,2,2)
        Y = 3 * numpy.random.rand(2,2,2,2)

        AX = UTPM(X)
        AY = UTPM(Y)

        assert_array_almost_equal( AX.dot(AY).tc, dot(AX,AY).tc)
        assert_array_almost_equal( AX.inv().tc,  inv(AX).tc)
        assert_array_almost_equal( AX.trace().tc,  trace(AX).tc)
        
    def test_operations_on_scalar_UTPM(self):
        D,P = 2,1
        X = 3 * numpy.random.rand(D,P)
        Y = 2 * numpy.random.rand(D,P)
        
        Z1 = numpy.zeros((D,P))
        Z2 = numpy.zeros((D,P))
        Z3 = numpy.zeros((D,P))
        Z4 = numpy.zeros((D,P))

        
        Z1[:,:] = X[:,:] + Y[:,:]
        Z2[:,:] = X[:,:] - Y[:,:]
        
        Z3[0,:] = X[0,:] * Y[0,:]
        Z3[1,:] = X[0,:] * Y[1,:] + X[1,:] * Y[0,:]
        
        Z4[0,:] = X[0,:] / Y[0,:]
        Z4[1,:] = 1./Y[0,:] * ( X[1,:] - X[0,:] * Y[1,:]/ Y[0,:])
        
        aX = UTPM(X)
        aY = UTPM(Y)
        
        aZ1 = aX + aY
        aZ2 = aX - aY
        aZ3 = aX * aY
        aZ4 = aX / aY
        aZ5 = aX.dot(aY)
        
        assert_array_almost_equal(aZ1.tc, Z1)
        assert_array_almost_equal(aZ2.tc, Z2)
        assert_array_almost_equal(aZ3.tc, Z3)
        assert_array_almost_equal(aZ4.tc, Z4)
        assert_array_almost_equal(aZ5.tc, Z3)
        
        
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
        AY4 = 2. / AX
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
        # Z4 = 1./X.copy()
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
        

    def test_constructor_stores_reference_of_tc_and_does_not_copy(self):
        X  = numpy.zeros((2,3,4,5))
        Y  = X + 1
        AX = UTPM(X)
        AX.tc[...] = 1.
        assert_array_almost_equal(AX.tc, Y)

    def test_getitem_single_element_of_vector(self):
        X  = numpy.zeros((2,3,4))
        X2 = X.copy()
        X2[:,:,0] += 1
        AX = UTPM(X)
        AY = AX[0]
        AY.tc[:,:] += 1
        assert_array_almost_equal(X,X2)

        
    def test_getitem_single_element_of_matrix(self):
        X  = numpy.zeros((2,3,4,5))
        X2 = X.copy()
        X2[:,:,0,0] += 1
        AX = UTPM(X)
        AY = AX[0,0]
        AY.tc[:,:] += 1
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
    
    def test_clone(self):
        D,P,N,M = 2,3,4,5
        X = 2 * numpy.random.rand(D,P,N,M)
        AX = UTPM(X)
        AY = AX.clone()
        
        AX.tc[...] += 13
        assert_equal(AY.tc.flags['OWNDATA'],True)
        assert_array_almost_equal( AX.tc, AY.tc + 13)
        
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
        
        for p in range(P):
            for n in range(N):
                A.data[0,p,n,n] += (N + 1)
        
        y = x.solve(A)
        x2 = A.dot(y)
        assert_array_almost_equal(x.tc, x2.tc, decimal = 12)

    def test_qr(self):
        (D,P,N) = 6,3,20
        A_data = numpy.random.rand(D,P,N,N)
        
        # make A_data sufficiently regular
        for p in range(P):
            for n in range(N):
                A_data[0,p,n,n] += (N + 1)
        A_data_old = A_data.copy()
        A = UTPM(A_data)

        Q,R = A.qr()
        assert_array_almost_equal( (Q.dot(R)).data, A_data_old, decimal = 12)
        
    def test_qr_rectangular(self):
        (D,P,M,N) = 5,3,5,3
        A_data = numpy.random.rand(D,P,M,N)
        
        # make A_data sufficiently regular
        for p in range(P):
            for n in range(N):
                A_data[0,p,n,n] += (N + 1)
        
        A = UTPM(A_data)

        Q,R = A.qr_rectangular()
        
        assert_array_equal( Q.data.shape, [D,P,M,N])
        assert_array_equal( R.data.shape, [D,P,N,N])
        
        # print 'zero?\n',dot(Q, R) - A
        assert_array_almost_equal( (Q.dot(R)).data, A.data, decimal = 14)


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
        

    def test_eig(self):
        (D,P,N) = 2,1,2
        A_data = numpy.zeros((D,P,N,N))
        for d in range(D):
            for p in range(P):
                tmp = numpy.random.rand(N,N)
                A_data[d,p,:,:] = numpy.dot(tmp.T,tmp)

        A = UTPM(A_data)
        L,Q = A.eig()
        
        # print L
        #print L,Q


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
