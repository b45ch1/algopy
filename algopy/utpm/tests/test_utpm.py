from numpy.testing import *
import numpy

from algopy.utpm import *

class Test_Push_Forward(TestCase):
    
    def test_as_utpm(self):
        D,P,N,M = 2,3,5,7
    
        A = numpy.zeros((3,4),dtype=object)
        for n in range(3):
            for m in range(4):
                A[n,m] = UTPM(numpy.random.rand(D,P,N,M))
        
        z = UTPM.as_utpm(A)
        
        
        for n in range(3):
            for m in range(4):
                assert_array_almost_equal(z[0,0].data, A[0,0].data)

    def test_sum(self):
        # check sum
        D,P,N,M = 2,3,4,5
        x = UTPM(numpy.arange(D*P*N*M).reshape((D,P,N,M)))
        
        y0 = numpy.sum(x).data
        y1 = numpy.sum(x,axis=0).data
        y2 = numpy.sum(x,axis=1).data
        
        for d in range(D):
            for p in range(P):
                assert_array_almost_equal(numpy.sum(x.data[d,p]), y0[d,p])
                assert_array_almost_equal(numpy.sum(x.data[d,p], axis=0), y1[d,p])
                assert_array_almost_equal(numpy.sum(x.data[d,p], axis=1), y2[d,p])
        

    def test_broadcasting_sub(self):
        #check 1
        x1 = numpy.array([1.,2.,3.])
        y1 = UTPM([[5],[7]])
        
        z11 = x1 - y1
        z12 = -(y1 - x1)
        
        z_data = numpy.array([[[-4,-3,-2]],[[-7,-7,-7]]])
        assert_array_almost_equal(z_data, z11.data)
        assert_array_almost_equal(z_data, z12.data)
        
        
    def test_broadcasting_setitem(self):
        x = UTPM(numpy.arange(2*1*3*4).reshape((2,1,3,4)))
        y = UTPM(numpy.arange(2*1).reshape((2,1)))
        x[0, :] = y
        x[:, 1] = y
        
        assert_array_almost_equal(0, x.data[0,:,0,:])
        assert_array_almost_equal(1, x.data[1,:,0,:])
        assert_array_almost_equal(0, x.data[0,:,:,1])
        assert_array_almost_equal(1, x.data[1,:,:,1])
        
        x[0, :] = 3.
        x[:, 1] = 3.
        
        assert_array_almost_equal(3, x.data[0,:,0,:])
        assert_array_almost_equal(0, x.data[1,:,0,:])
        assert_array_almost_equal(3, x.data[0,:,:,1])
        assert_array_almost_equal(0, x.data[1,:,:,1])

    

                
    def test_symvec_vecsym(self):
        (D,P,N) = 2,1,5
        A = UTPM(numpy.random.rand(*(D,P,N,N)))
        A = UTPM.dot(A.T,A)
        v = UTPM.symvec(A)
        B = UTPM.vecsym(v)
        
        assert_array_almost_equal(A.data, B.data)
        
        
    def test_symvec_vecsym_pullback(self):
        (D,P,N) = 2,1,6
        v = UTPM(numpy.random.rand(*(D,P,N)))
        A = UTPM.vecsym(v)
        w = UTPM.symvec(A)
        wbar = UTPM(numpy.random.rand(*(D,P,N)))
        Abar = UTPM.pb_symvec(wbar, A, w)
        vbar = UTPM.pb_vecsym(Abar, v, A)
        
        assert_array_almost_equal(wbar.data, vbar.data)
    
    
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
        AZ = UTPM.dot(AX,AY)
        AZ = UTPM.inv(AX)
        AZ = UTPM.trace(AX)
        AZ = AX.T
        AX = AX.set_zero()


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
        aZ5 = UTPM.dot(aX,aY)

        assert_array_almost_equal(aZ1.data, Z1)
        assert_array_almost_equal(aZ2.data, Z2)
        assert_array_almost_equal(aZ3.data, Z3)
        assert_array_almost_equal(aZ4.data, Z4)
        assert_array_almost_equal(aZ5.data, Z3)


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

        assert_array_equal( UTPM.dot(aX,aY).data.shape, (D,P,N,N))
        assert_array_equal( UTPM.dot(aY,aX).data.shape, (D,P,M,M))
        assert_array_equal( UTPM.dot(aA,ax).data.shape, (D,P,M))
        assert_array_equal( UTPM.dot(ax,ay).data.shape, (D,P))


    def test_dot_non_UTPM(self):
        D,P,N,M = 2,3,4,5
        RX = 2 * numpy.random.rand(D,P,N,M)
        RY = 3 * numpy.random.rand(D,P,M,N)
        RA = 3 * numpy.random.rand(D,P,M,N)
        Rx = 2 * numpy.random.rand(D,P,N)
        Ry = 2 * numpy.random.rand(D,P,N)

        X = RX[0,0]
        Y = RY[0,0]
        A = RA[0,0]
        x = Rx[0,0]
        y = Ry[0,0]

        aX = UTPM(RX)
        aY = UTPM(RY)
        aA = UTPM(RA)
        ax = UTPM(Rx)
        ay = UTPM(Ry)

        assert_array_almost_equal(UTPM.dot(aX,aY).data[0,0], UTPM.dot(aX, Y).data[0,0])
        assert_array_almost_equal(UTPM.dot(aX,aY).data[0,0], UTPM.dot(X, aY).data[0,0])

        assert_array_almost_equal(UTPM.dot(aA,ax).data[0,0], UTPM.dot(aA, x).data[0,0])
        assert_array_almost_equal(UTPM.dot(aA,ax).data[0,0], UTPM.dot(A, ax).data[0,0])

        assert_array_almost_equal(UTPM.dot(aA,aX).data[0,0], UTPM.dot(aA, X).data[0,0])
        assert_array_almost_equal(UTPM.dot(aA,aX).data[0,0], UTPM.dot(A, aX).data[0,0])

        assert_array_almost_equal(UTPM.dot(ax,ay).data[0,0], UTPM.dot(ax, y).data[0,0])
        assert_array_almost_equal(UTPM.dot(ax,ay).data[0,0], UTPM.dot(x, ay).data[0,0])



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

        assert_array_almost_equal(AY1.data, Z1 )
        assert_array_almost_equal(AY2.data, Z2 )
        assert_array_almost_equal(AY3.data, Z3 )
        # assert_array_almost_equal(AY4.data, Z4 )
        assert_array_almost_equal(AY5.data, Z5 )
        assert_array_almost_equal(AY6.data, Z6 )
        assert_array_almost_equal(AY7.data, Z7 )
        assert_array_almost_equal(AY8.data, Z8 )

        assert_array_almost_equal(AX1.data, AY5.data )
        assert_array_almost_equal(AX2.data, AY6.data )
        assert_array_almost_equal(AX3.data, AY7.data )
        assert_array_almost_equal(AX4.data, AY8.data )


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

        assert_array_almost_equal(AX1.data, AY5.data )
        assert_array_almost_equal(AX2.data, AY6.data )
        assert_array_almost_equal(AX3.data, AY7.data )
        assert_array_almost_equal(AX4.data, AY8.data )
        
    def test_sqrt(self):
        D,P,N = 5,3,2
        X = UTPM(numpy.random.rand(D,P,N,N))
        Y = UTPM.sqrt(X)
        assert_array_almost_equal(X.data, (Y*Y).data)

    def test_exp_log(self):
        D,P,N = 4,2,2
        X = UTPM(numpy.random.rand(D,P,N,N))
        X.data[1] = 1.
        Y = UTPM.exp(X)
        Z = UTPM.log(Y)
        assert_array_almost_equal(X.data, Z.data)
        
    def test_pow(self):
        D,P,N = 4,2,2
        X = UTPM(numpy.random.rand(D,P,N,N))
        Y = X**3
        Z = X*X*X
        assert_array_almost_equal(Y.data, Z.data)
        
        
    def test_pow_pullback(self):
        D,P,N = 4,2,2
        x = UTPM(numpy.random.rand(D,P,N))
        
        r = 2
        y = x**r
        ybar = UTPM(numpy.random.rand(D,P,N))
        xbar = UTPM.pb___pow__(ybar, x, r, y)
        
        assert_array_almost_equal((r * ybar * x**(r-1)).data, xbar.data)
        
        
        r = 3.1
        y = x**r
        ybar = UTPM(numpy.random.rand(D,P,N))
        xbar = UTPM.pb___pow__(ybar, x, r, y)
        
        assert_array_almost_equal((r * ybar * x**(r-1)).data, xbar.data)
        
    def test_sincos(self):
        D,P,M,N = 4,3,2,1
        X = UTPM(numpy.random.rand(D,P,M,N))
        Z = UTPM.sin(X)**2 + UTPM.cos(X)**2
        # print Z
        assert_array_almost_equal(Z.data[0], numpy.ones((P,M,N)))
        assert_array_almost_equal(Z.data[1], numpy.zeros((P,M,N)))

    def test_tansec(self):
        D,P,M,N = 4,3,2,1
        
        x = UTPM(numpy.random.random((D,P,M,N)))
        y1 = UTPM.tan(x)
        y2 = UTPM.sin(x)/UTPM.cos(x)
        assert_array_almost_equal(y2.data, y1.data)
        
    def test_arcsin(self):
        D,P,N,M = 5,3,4,5
        x = UTPM(numpy.random.random((D,P,M,N)))
        
        y = UTPM.arcsin(x)
        x2 = UTPM.sin(y)
        y2 = UTPM.arcsin(x2)
        
        assert_array_almost_equal(x.data, x2.data)
        assert_array_almost_equal(y.data, y2.data)        


     
    def test_abs(self):
        D,P,N = 4,3,12
        tmp = numpy.random.rand(D,P,N,N) - 0.5
        tmp[0,:] = tmp[0,0]
        
        X = UTPM(tmp)
        assert_array_almost_equal(X.data, 0.5*( abs(X + abs(X)) - abs(X - abs(X))).data)
        
    def test_shift(self):
        D,P,N = 5,1,2
        X = UTPM(numpy.random.rand(D,P,N))
        Y = UTPM.shift(X,2)
        assert_array_almost_equal(X.data[:-2,...], Y.data[2:,...])
        
        Z = UTPM.shift(X,-2)
        assert_array_almost_equal(X.data[2:,...], Z.data[:-2,...])

    def test_max(self):
        D,P,N = 2,3,4
        X = numpy.array([ dpn for dpn in range(D*P*N)],dtype = float)
        X = X.reshape((D,P,N))
        AX = UTPM(X)
        axmax = UTPM.max(AX)
        #print axmax
        #print  AX.data[:,:,-1]
        assert_array_almost_equal(axmax.data, AX.data[:,:,-1])

    def test_argmax(self):
        D,P,N,M = 2,3,4,5
        X = numpy.array([ dpn for dpn in range(D*P*N*M)],dtype = float)
        X = X.reshape((D,P,N,M))
        AX = UTPM(X)
        amax = UTPM.argmax(AX)
        assert_array_equal(amax, [19,19,19])
        

    def test_constructor_stores_reference_of_tc_and_does_not_copy(self):
        X  = numpy.zeros((2,3,4,5))
        Y  = X + 1
        AX = UTPM(X)
        AX.data[...] = 1.
        assert_array_almost_equal(AX.data, Y)

    def test_getitem_single_element_of_vector(self):
        X  = numpy.zeros((2,3,4))
        X2 = X.copy()
        X2[:,:,0] += 1
        AX = UTPM(X)
        AY = AX[0]
        AY.data[:,:] += 1
        assert_array_almost_equal(X,X2)


    def test_getitem_single_element_of_matrix(self):
        X  = numpy.zeros((2,3,4,5))
        X2 = X.copy()
        X2[:,:,0,0] += 1
        AX = UTPM(X)
        AY = AX[0,0]
        AY.data[:,:] += 1
        assert_array_almost_equal(X,X2)

    def test_getitem_slice(self):
        X  = numpy.zeros((2,3,4,5))
        X2 = X.copy()
        X2[:,:,0:2,1:3] += 1
        AX = UTPM(X)
        AY = AX[0:2,1:3]
        AY.data[:,:] += 1.
        assert_array_almost_equal(X,X2)
     
    def test_getitem_of_2D_array(self):
        D,P,N = 2,3,7
        ax = UTPM(numpy.random.rand(D,P,N,N))
        
        for r in range(N):
            for c in range(N):
                assert_array_almost_equal(ax[r,c].data, ax.data[:,:,r,c])
                
    def test_getitem_slice(self):
        D,P,N = 1,1,10
        ax = UTPM(numpy.random.rand(D,P,N))
        tmp = ax[:N//2]
        tmp += 1.
        
        assert_array_almost_equal(ax.data[0,0,:N//2], tmp.data[0,0,:])
        
    def test_setitem_slice(self):
        D,P,N = 1,1,10
        ax_data = numpy.random.rand(D,P,N)
        ax_data2 = ax_data.copy()
        ax = UTPM(ax_data)
        ax[:N//2] += 1
        
        ax_data2[:,:,:N//2] += 1
        assert_array_almost_equal(ax.data[:,:,:N//2], ax_data2[:,:,:N//2])
        
    def test_setitem_with_scalar(self):
        D,P,N = 3,2,5
        ax_data = numpy.ones((D,P,N))
        ax = UTPM(ax_data)
        ax[...] = 1.2
        
        assert_array_almost_equal(ax.data[0,...], 1.2)
        assert_array_almost_equal(ax.data[1:,...], 0)

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

        AX.data[...] += 13
        assert_equal(AY.data.flags['OWNDATA'],True)
        assert_array_almost_equal( AX.data, AY.data + 13)

    def test_reshape(self):
        D,P,N,M = 2,3,4,5
        X  = numpy.zeros((D,P,N,M))
        AX = UTPM(X)
        AY = UTPM.reshape(AX, (5,4))
        assert_array_equal(AY.data.shape, (2,3,5,4))
        assert AY.data.flags['OWNDATA']==False


    def test_transpose(self):
        D,P,N,M = 2,3,4,5
        X  = UTPM(numpy.random.rand(*(D,P,N,M)))
        Y = X.T

        Y.data[0,0,1,0] += 123
        Z = Y.T
        assert_array_equal(Y.data.shape, (D,P,M,N))

        #check that no copy is made
        assert_array_almost_equal(Z.data, X.data)



    def test_diag(self):
        D,P,N = 2,3,4
        x = UTPM(numpy.random.rand(D,P,N))

        X = UTPM.diag(x)

        for n in range(N):
            assert_almost_equal( x.data[...,n], X.data[...,n,n])

    def test_diag_pullback(self):
        D,P,N = 2,3,4
        # forward
        x = UTPM(numpy.random.rand(D,P,N))
        X = UTPM.diag(x)
        
        #reverse
        Xbar = UTPM.diag(UTPM(numpy.random.rand(D,P,N)))
        xbar = UTPM.pb_diag(Xbar, x, X)
        
        assert_array_almost_equal(UTPM.diag(Xbar).data,xbar.data)


    

    def test_trace(self):
        N1 = 2
        N2 = 3
        N3 = 4
        N4 = 5
        x = numpy.asarray(range(N1*N2*N3*N4))
        x = x.reshape((N1,N2,N3,N4))
        AX = UTPM(x)
        AY = AX.T
        AY.data[0,0,2,0] = 1234
        assert AX.data[0,0,0,2] == AY.data[0,0,2,0]

    def test_inv(self):
        (D,P,N,M) = 2,3,5,1
        A = UTPM(numpy.random.rand(D,P,N,N))
        Ainv = UTPM.inv(A)

        Id = numpy.zeros((D,P,N,N))
        Id[0,:,:,:] = numpy.eye(N)
        assert_array_almost_equal(UTPM.dot(A, Ainv).data, Id)

    def test_solve(self):
        (D,P,N,M) = 3,3,30,5
        x = UTPM(numpy.random.rand(D,P,N,M))
        A = UTPM(numpy.random.rand(D,P,N,N))

        for p in range(P):
            for n in range(N):
                A.data[0,p,n,n] += (N + 1)

        y = UTPM.solve(A,x)
        x2 = UTPM.dot(A, y)
        assert_array_almost_equal(x.data, x2.data, decimal = 12)

    def test_solve_non_UTPM_x(self):
        (D,P,N) = 2,3,2
        A  = UTPM(numpy.random.rand(D,P,N,N))
        Id = numpy.zeros((N,N))

        for p in range(P):
            for n in range(N):
                A[n,n] += (N + 1)
                Id[n,n] = 1

        y = UTPM.solve(A,Id)
        Id2 = UTPM.dot(A, y)

        for p in range(P):
            assert_array_almost_equal(Id, Id2.data[0,p], decimal = 12)

        assert_array_almost_equal(numpy.zeros((D-1,P,N,N)), Id2.data[1:], decimal=10)

    def test_solve_non_UTPM_x(self):
        """
        check that Id Y = X yields Y = inv(X)
        """
        (D,P,N) = 2,3,2
        Id = numpy.eye(N)
        X  = UTPM(numpy.random.rand(D,P,N,N))

        Y  = UTPM.solve(X,Id)
        Y2 = UTPM.inv(X)
        
        assert_array_almost_equal(Y.data, Y2.data)



    def test_shape(self):
        D,P,N,M,L = 3,4,5,6,7
        
        x = UTPM(numpy.random.rand(D,P,N))
        y = UTPM(numpy.random.rand(D,P,N,M))
        z = UTPM(numpy.random.rand(D,P,N,M))

        #UTPM.shape(x)
        
        
    def test_iouter(self):
        D,P,N = 3,4,5
        x = UTPM(numpy.random.rand(D,P,N))
        y = UTPM(numpy.random.rand(D,P,N))
        z = UTPM(numpy.random.rand(D,P,N))

        A = UTPM(numpy.random.rand(*(D,P,N,N)))
        B = UTPM(A.data.copy())

        UTPM.iouter(x,y,A)

        r1 = UTPM.dot(A,z)
        r2 = UTPM.dot(B, z) + x * UTPM.dot(y,z)

        assert_array_almost_equal(r2.data, r1.data)


class Test_Pullbacks(TestCase):
    def test_solve_pullback(self):
        (D,P,N,K) = 2,5,3,4
        A = UTPM(numpy.random.rand(D,P,N,N))
        x = UTPM(numpy.random.rand(D,P,N,K))
        
        y = UTPM.solve(A,x)

        assert_array_almost_equal( x.data, UTPM.dot(A,y).data)
        
        ybar = UTPM(numpy.random.rand(*y.data.shape))
        Abar, xbar = UTPM.pb_solve(ybar, A, x, y)

        for p in range(P):
            Ab = Abar.data[0,p]
            Ad = A.data[1,p]

            xb = xbar.data[0,p]
            xd = x.data[1,p]

            yb = ybar.data[0,p]
            yd = y.data[1,p]

            assert_almost_equal( numpy.trace(numpy.dot(Ab.T,Ad)) + numpy.trace(numpy.dot(xb.T,xd)), numpy.trace(numpy.dot(yb.T,yd)))
            
            
    # def test_dot_pullback(self):
    #     import adolc
    #     import adolc.cgraph
        
    #     D,P,N,M,L = 3,4,5,6,7
    #     A = numpy.random.rand(*(N,M))
    #     B = numpy.random.rand(*(M,L))
        
    #     cg = adolc.cgraph.AdolcProgram()
    #     cg.trace_on(1)
    #     aA = adolc.adouble(A)
    #     aB = adolc.adouble(B)
        
    #     cg.independent(aA)
    #     cg.independent(aB)
        
    #     aC = numpy.dot(aA, aB)
        
    #     cg.dependent(aC)
    #     cg.trace_off()
        
    #     VA = numpy.random.rand(N,M,P,D-1)
    #     VB = numpy.random.rand(M,L,P,D-1)
        
    #     cg.forward([A,B],[VA,VB])
        
    #     WC = numpy.random.rand(1, N,L,P,D)
    #     WA, WB = cg.reverse([WC])
        
    #     # print WA,WB
    #     # assert False
        
    def test_dot_pullback(self):
        D,P,N,K,M = 3,4,5,6,7
        X = UTPM(numpy.random.rand(D,P,N,K))
        Y = UTPM(numpy.random.rand(D,P,K,M))
        
        Z = UTPM.dot(X,Y)
        Zbar = UTPM(numpy.random.rand(D,P,N,M))
        
        Xbar, Ybar = UTPM.pb_dot(Zbar, X, Y, Z)
       
        Xbar2 = UTPM.dot(Zbar, Y.T)
        Ybar2 = UTPM.dot(X.T, Zbar)
        
        assert_array_almost_equal(Xbar2.data, Xbar.data)
        
    def test_inv_pullback(self):
        D,P,N = 3,4,5
        X = UTPM(numpy.random.rand(D,P,N,N))
        
        #make X sufficiently well-conditioned
        for n in range(N):
            X[n,n] += N+1.
        
        Ybar = UTPM(numpy.random.rand(D,P,N,N))
        
        Y = UTPM.inv(X)
        
        Xbar = UTPM.pb_inv(Ybar, X, Y)
        
        Xbar2 = -1*UTPM.dot(UTPM.dot(Y.T, Ybar), Y.T)
        assert_array_almost_equal(Xbar.data, Xbar2.data, decimal=12)
        
    def test_inv_trace_pullback(self):
        (D,P,M,N) = 3,9,3,3
        A = UTPM(numpy.zeros((D,P,M,M)))
        
        A0 = numpy.random.rand(M,N)
        
        for m in range(M):
            for n in range(N):
                p = m*N + n
                A.data[0,p,:M,:N] = A0
                A.data[1,p,m,n] = 1.

        B = UTPM.inv(A)
        y = UTPM.trace(B)
        ybar = y.zeros_like()
        ybar.data[0,:] = 1.
        
        Bbar = UTPM.pb_trace(ybar, B, y)
        Abar = UTPM.pb_inv(Bbar, A, B)
        
        assert_array_almost_equal(Abar.data[0,0].ravel(), y.data[1])
        
        
        tmp = []
        for m in range(M):
            for n in range(N):
                p = m*N + n
                tmp.append( Abar.data[1,p,m,n])
        
        assert_array_almost_equal(tmp, 2*y.data[2])
        

    def test_pullback_solve_for_inversion(self):
        """
        test pullback on
        x = solve(A,Id)
        """
    
        (D,P,N) = 2,7,10
        A_data = numpy.random.rand(D,P,N,N)
        
        # make A_data sufficiently regular
        for p in range(P):
            for n in range(N):
                A_data[0,p,n,n] += (N + 1)
        
        A = UTPM(A_data)
        
        # method 1: computation of the inverse matrix by solving an extended linear system
        # forward
        Id = numpy.eye(N)
        x = UTPM.solve(A,Id)
        # reverse
        xbar = UTPM(numpy.random.rand(D,P,N,N))
        Abar1, Idbar = UTPM.pb_solve(xbar, A, Id, x)
        
        # method 2: direct inversion
        # forward
        Ainv = UTPM.inv(A)
        # reverse
        Abar2 = UTPM.pb_inv(xbar, A, Ainv)
        
        assert_array_almost_equal(x.data, Ainv.data)
        assert_array_almost_equal(Abar1.data, Abar2.data)


class Test_Cholesky_Decomposition(TestCase):
    def test_pushforward(self):
        D,P,N = 5, 2, 10
        tmp = numpy.random.rand(*(D,P,N,N))
        A = UTPM(tmp)
        A = UTPM.dot(A.T,A)

        L = UTPM.cholesky(A)
        assert_array_almost_equal( A.data, UTPM.dot(L,L.T).data)
        
        

class Test_QR_Decomposition(TestCase):
    def test_pushforward(self):
        (D,P,N) = 3,5,10
        A_data = numpy.random.rand(D,P,N,N)

        # make A_data sufficiently regular
        for p in range(P):
            for n in range(N):
                A_data[0,p,n,n] += (N + 1)
        A_data_old = A_data.copy()
        A = UTPM(A_data)

        Q,R = UTPM.qr(A)
        assert_array_almost_equal(UTPM.triu(R).data,  R.data)
        assert_array_almost_equal( ( UTPM.dot(Q,R)).data, A_data_old, decimal = 12)
        assert_array_almost_equal(UTPM.dot(Q.T,Q).data[0], [numpy.eye(N) for p in range(P)])
        assert_array_almost_equal(UTPM.dot(Q.T,Q).data[1:],0)


    def test_pushforward_rectangular_A_qr_full(self):
        D,P,M,N = 5,3,4,2
        A = UTPM(numpy.random.rand(D,P,M,N))
        Q,R = UTPM.qr_full(A)
        
        assert_array_almost_equal(UTPM.triu(R).data,  R.data)
        assert_array_almost_equal(A.data, UTPM.dot(Q,R).data)
        assert_array_almost_equal(0, (UTPM.dot(Q.T,Q) - numpy.eye(M)).data)


    def test_pullback_rectangular_A_qr_full(self):
        D,P,M,N = 2,9,30,10
    
        # forward
        A = UTPM(numpy.random.rand(D,P,M,N))
        Q,R = UTPM.qr_full(A)
        A2 = UTPM.dot(Q,R)
        Q2, R2 = UTPM.qr_full(A2)
        
        # reverse
        Q2bar = UTPM(numpy.random.rand(D,P,M,M))
        R2bar = UTPM.triu(UTPM(numpy.random.rand(D,P,M,N)))
        
        A2bar = UTPM.pb_qr_full(Q2bar, R2bar, A2, Q2, R2)
        Qbar, Rbar = UTPM.pb_dot(A2bar, Q, R, A2)
        Abar = UTPM.pb_qr_full(Qbar, Rbar, A, Q, R)
        
        # check forward calculation
        assert_array_almost_equal(Q.data, Q2.data)
        assert_array_almost_equal(R.data, R2.data)
        
        # check reverse calculation: PART I
        assert_array_almost_equal(Abar.data, A2bar.data)
        assert_array_almost_equal( UTPM.triu(Rbar).data,  UTPM.triu(R2bar).data)
        # cannot check Qbar and Q2bar since Q has only N*M - N(N+1)/2 distinct elements
        
        
        # check reverse calculation: PART II
        for p in range(P):
            Ab = Abar.data[0,p]
            Ad = A.data[1,p]
        
            Q2b = Q2bar.data[0,p]
            Q2d = Q2.data[1,p]
        
            R2b = R2bar.data[0,p]
            R2d = R2.data[1,p]
        
            assert_almost_equal(numpy.trace(numpy.dot(Ab.T,Ad)), numpy.trace(numpy.dot(Q2b.T,Q2d)) + numpy.trace(numpy.dot(R2b.T,R2d)))

    def test_pushforward_rectangular_A(self):
        (D,P,M,N) = 5,3,15,3
        A_data = numpy.random.rand(D,P,M,N)

        # make A_data sufficiently regular
        for p in range(P):
            for n in range(N):
                A_data[0,p,n,n] += (N + 1)

        A = UTPM(A_data)

        Q,R = UTPM.qr(A)

        assert_array_equal( Q.data.shape, [D,P,M,N])
        assert_array_equal( R.data.shape, [D,P,N,N])

        # print 'zero?\n',dot(Q, R) - A
        assert_array_almost_equal(UTPM.triu(R).data,  R.data)
        assert_array_almost_equal( (UTPM.dot(Q,R)).data, A.data, decimal = 14)
        assert_array_almost_equal(UTPM.dot(Q.T,Q).data[0], [numpy.eye(N) for p in range(P)])
        assert_array_almost_equal(UTPM.dot(Q.T,Q).data[1:],0)
        
    def test_singular_matrix1(self):
        D,P,M,N = 3,1,40,20
        A = UTPM(numpy.random.rand(D,P,M,M))
        A[N:,:] = 0
        Q,R = UTPM.qr(A)
        
        assert_array_almost_equal(UTPM.triu(R).data,  R.data)
        assert_array_almost_equal(A.data, UTPM.dot(Q,R).data)
        assert_array_almost_equal(0, (UTPM.dot(Q.T,Q) - numpy.eye(M)).data)
        
        
    def test_singular_matrix2(self):
        D,P,M,N = 3,1,40,20
        x = UTPM(numpy.random.rand(D,P,M,N))
        A = UTPM.dot(x,x.T)
        Q,R = UTPM.qr(A)
        
        assert_array_almost_equal(UTPM.triu(R).data,  R.data)
        assert_array_almost_equal(A.data, UTPM.dot(Q,R).data)
        assert_array_almost_equal(0, (UTPM.dot(Q.T,Q) - numpy.eye(M)).data)
        
        # check that columns of Q2 span the nullspace of A
        Q2 = Q[:,N:]
        assert_array_almost_equal(0, UTPM.dot(A.T, Q2).data)

    def test_singular_matrix3(self):
        D,P,M,N = 3,1,40,20
        A = UTPM(numpy.random.rand(D,P,M,M))
        A[:,N:] = 0
        Q,R = UTPM.qr(A)
        
        assert_array_almost_equal(UTPM.triu(R).data,  R.data)
        assert_array_almost_equal(A.data, UTPM.dot(Q,R).data)
        assert_array_almost_equal(0, (UTPM.dot(Q.T,Q) - numpy.eye(M)).data)
        
        # check that columns of Q2 span the nullspace of A
        Q2 = Q[:,N:]
        assert_array_almost_equal(0, UTPM.dot(A.T, Q2).data)
        
    def test_pushforward_more_cols_than_rows(self):
        """
        A.shape = (3,11)
        """
        (D,P,M,N) = 5,3,2,15
        A_data = numpy.random.rand(D,P,M,N)
        
        # make A_data sufficiently regular
        for p in range(P):
            for m in range(M):
                A_data[0,p,m,m] += (N + 1)
        

        A = UTPM(A_data)

        Q,R = UTPM.qr(A)

        assert_array_equal( Q.data.shape, [D,P,M,M])
        assert_array_equal( R.data.shape, [D,P,M,N])

        # print 'zero?\n',dot(Q, R) - A
        # assert_array_almost_equal( (UTPM.dot(Q,R[:,:M])).data, A[:,:M].data, decimal = 14)
        # assert_array_almost_equal( (UTPM.dot(Q,R[:,M:])).data, A[:,M:].data, decimal = 14)
        assert_array_almost_equal(UTPM.triu(R).data,  R.data)
        assert_array_almost_equal( (UTPM.dot(Q,R)).data, A.data, decimal = 12)
        assert_array_almost_equal(UTPM.dot(Q.T,Q).data[0], [numpy.eye(M) for p in range(P)])
        assert_array_almost_equal(UTPM.dot(Q.T,Q).data[1:],0)

    def test_pullback(self):
        (D,P,M,N) = 2,3,10,10

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
                Rbar_data[:,:,r,c] *= (c>r)


        Qbar = UTPM(Qbar_data)
        Rbar = UTPM(Rbar_data)

        Abar = UTPM.pb_qr(Qbar, Rbar, A, Q, R)
        for p in range(P):
            Ab = Abar.data[0,p]
            Ad = A.data[1,p]

            Qb = Qbar.data[0,p]
            Qd = Q.data[1,p]

            Rb = Rbar.data[0,p]
            Rd = R.data[1,p]
            assert_almost_equal(numpy.trace(numpy.dot(Ab.T,Ad)), numpy.trace(numpy.dot(Qb.T,Qd) + numpy.dot(Rb.T,Rd)))

    def test_pullback_rectangular_A(self):
        (D,P,M,N) = 2,7,10,3

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
                Rbar_data[:,:,r,c] *= (c>r)


        Qbar = UTPM(Qbar_data)
        Rbar = UTPM(Rbar_data)

        Abar = UTPM.pb_qr(Qbar, Rbar, A, Q, R)

        for p in range(P):
            Ab = Abar.data[0,p]
            Ad = A.data[1,p]

            Qb = Qbar.data[0,p]
            Qd = Q.data[1,p]

            Rb = Rbar.data[0,p]
            Rd = R.data[1,p]
            assert_almost_equal(numpy.trace(numpy.dot(Ab.T,Ad)), numpy.trace(numpy.dot(Qb.T,Qd) + numpy.dot(Rb.T,Rd)))

    def test_pullback_more_cols_than_rows(self):
        (D,P,M,N) = 3,3,5,17
        A_data = numpy.random.rand(D,P,M,N)
        
        A = UTPM(A_data)
        Q,R = UTPM.qr(A)
        
        Qbar = UTPM(numpy.random.rand(D,P,M,M))
        Rbar = UTPM(numpy.random.rand(D,P,M,N))
        for r in range(M):
            for c in range(N):
                Rbar[r,c] *= (c>=r)

        Abar = UTPM.pb_qr(Qbar, Rbar, A, Q, R)
        
        for p in range(P):
            Ab = Abar.data[0,p]
            Ad = A.data[1,p]
            Qb = Qbar.data[0,p]
            Qd = Q.data[1,p]
            Rb = Rbar.data[0,p]
            Rd = R.data[1,p]
            
            assert_almost_equal(numpy.trace(numpy.dot(Ab.T,Ad)), numpy.trace(numpy.dot(Qb.T,Qd)) + numpy.trace( numpy.dot(Rb.T,Rd)))


    def test_pullback_singular_matrix(self):
        D,P,M,N = 2,1,6,3
        A = UTPM(numpy.zeros((D,P,M,M)))
        A[:,:N] = numpy.random.rand(M,N)
        
        for m in range(M):
            for n in range(N):
                A.data[1] = 0.
                A.data[1,0,m,n] = 1.
                
                # STEP1: forward
                Q,R = UTPM.qr(A)
                B = UTPM.dot(Q,R)
                y = UTPM.trace(B)
                
                # STEP2: reverse
                ybar = y.zeros_like()
                ybar.data[0,0] = 13./7.
                Bbar = UTPM.pb_trace(ybar, B, y)
                Qbar, Rbar = UTPM.pb_dot(Bbar, Q, R, B)
                Abar = UTPM.pb_qr(Qbar, Rbar, A, Q, R)
                
                assert_array_almost_equal((m==n)*13./7., Abar.data[0,0,m,n])

    def test_UTPM_and_array(self):
        D,P,N = 2,2,2
        x = 2 * numpy.random.rand(D,P,N,N)
        y = 3 * numpy.random.rand(D,P,N,N)
        
        ax = UTPM(x)
        
        az11 = ax + y[0,0]
        az12 = ax - y[0,0]
        az13 = ax * y[0,0]
        az14 = ax / y[0,0]
        
        az21 = y[0,0] + ax
        az22 = - (y[0,0] - ax)
        az23 = y[0,0]*ax
        az24 = 1./( y[0,0]/ax)
        
        cz1 = x.copy()
        for p in range(P):
            cz1[0,p] += y[0,0]
            
        cz2 = x.copy()
        for p in range(P):
            cz2[0,p] -= y[0,0]

        cz3 = x.copy()
        for d in range(D):
            for p in range(P):
                cz3[d,p] *= y[0,0]
                
        cz4 = x.copy()
        for d in range(D):
            for p in range(P):
                cz4[d,p] /= y[0,0]
            
        assert_array_almost_equal(az11.data, cz1)
        assert_array_almost_equal(az21.data, cz1)
        
        assert_array_almost_equal(az12.data, cz2)
        assert_array_almost_equal(az22.data, cz2)
        
        assert_array_almost_equal(az13.data, cz3)
        assert_array_almost_equal(az23.data, cz3)
        
        assert_array_almost_equal(az14.data, cz4)
        assert_array_almost_equal(az24.data, cz4)
        
    def test_UTPM_and_scalar(self):
        D,P,N = 2,2,2
        x = 2 * numpy.random.rand(D,P,N,N)
        y = 3
        
        ax = UTPM(x)
        
        az11 = ax + y
        az12 = ax - y
        az13 = ax * y
        az14 = ax / y
        
        az21 = y + ax
        az22 = - (y - ax)
        az23 = y*ax
        az24 = 1/( y/ax)
        
        cz1 = x.copy()
        for p in range(P):
            cz1[0,p] += y
            
        cz2 = x.copy()
        for p in range(P):
            cz2[0,p] -= y

        cz3 = x.copy()
        for d in range(D):
            for p in range(P):
                cz3[d,p] *= y
                
        cz4 = x.copy()
        for d in range(D):
            for p in range(P):
                cz4[d,p] /= y
            
        assert_array_almost_equal(az11.data, cz1)
        assert_array_almost_equal(az21.data, cz1)
        
        assert_array_almost_equal(az12.data, cz2)
        assert_array_almost_equal(az22.data, cz2)
        
        assert_array_almost_equal(az13.data, cz3)
        assert_array_almost_equal(az23.data, cz3)
        
        assert_array_almost_equal(az14.data, cz4)
        assert_array_almost_equal(az24.data, cz4)

        
class Test_Eigenvalue_Decomposition(TestCase):
    
    def test_eigh1_pushforward(self):
        (D,P,N) = 2,1,2
        A = UTPM(numpy.zeros((D,P,N,N)))
        A.data[0,0] = numpy.eye(N)
        A.data[1,0] = numpy.diag([3,4])
        
        L,Q,b = UTPM.eigh1(A)
        
        assert_array_almost_equal(UTPM.dot(Q, UTPM.dot(L,Q.T)).data, A.data, decimal = 13)
        
        Lbar = UTPM.diag(UTPM(numpy.zeros((D,P,N))))
        Lbar.data[0,0] = [0.5,0.5]
        Qbar = UTPM(numpy.random.rand(*(D,P,N,N)))
        
        Abar = UTPM.pb_eigh1( Lbar, Qbar, None, A, L, Q, b)
        
        Abar = Abar.data[0,0]
        Adot = A.data[1,0]
        
        Lbar = Lbar.data[0,0]
        Ldot = L.data[1,0]
        
        Qbar = Qbar.data[0,0]
        Qdot = Q.data[1,0]
        
        assert_almost_equal(numpy.trace(numpy.dot(Abar.T, Adot)), numpy.trace( numpy.dot(Lbar.T, Ldot) + numpy.dot(Qbar.T, Qdot)))    
    

    def test_pushforward(self):
        (D,P,N) = 3,2,5
        A_data = numpy.zeros((D,P,N,N))
        for d in range(D):
            for p in range(P):
                tmp = numpy.random.rand(N,N)
                A_data[d,p,:,:] = numpy.dot(tmp.T,tmp)

                if d == 0:
                    A_data[d,p,:,:] += N * numpy.diag([n+1 for n in range(N)])

        A = UTPM(A_data)
        l,Q = UTPM.eigh(A)
        
        L = UTPM.diag(l)

        assert_array_almost_equal(UTPM.dot(Q, UTPM.dot(L,Q.T)).data, A.data, decimal = 12)

    def test_pushforward_repeated_eigenvalues(self):
        D,P,N = 3,1,6
        A = UTPM(numpy.zeros((D,P,N,N)))
        V = UTPM(numpy.random.rand(D,P,N,N))
        
        A.data[0,0] = numpy.diag([2,2,3,3.,4,5])
        A.data[1,0] = numpy.diag([5,1,3,1.,1,3])
        
        V,Rtilde = UTPM.qr(V)
        A = UTPM.dot(UTPM.dot(V.T, A), V)

        l,Q = UTPM.eigh(A)
        L = UTPM.diag(l)
        
        # print l

        assert_array_almost_equal(UTPM.dot(Q, UTPM.dot(L,Q.T)).data, A.data, decimal = 12)

    def test_pushforward_repeated_eigenvalues_higher_order_multiple_direction(self):
        D,P,N = 4,7,6
        A = UTPM(numpy.zeros((D,P,N,N)))
        V = UTPM(numpy.random.rand(D,P,N,N))
        
        A.data[0,0] = numpy.diag([2,2,2,3.,3.,3.])
        A.data[1,0] = numpy.diag([1,1,3,2.,2,2])
        A.data[2,0] = numpy.diag([7,5,5,1.,2,2])
        
        
        V,Rtilde = UTPM.qr(V)
        A = UTPM.dot(UTPM.dot(V.T, A), V)

        l,Q = UTPM.eigh(A)
        L = UTPM.diag(l)
        
        # for d in range(D):
        #     print l.data[d,0]
        #     print numpy.diag(UTPM.dot(Q.T, UTPM.dot(A,Q)).data[d,0])\
            
        # print UTPM.dot(Q.T, UTPM.dot(A,Q)).data

        assert_array_almost_equal(UTPM.dot(Q.T, UTPM.dot(A,Q)).data, L.data)    


    def test_pullback(self):
        (D,P,N) = 2,5,10
        A_data = numpy.zeros((D,P,N,N))
        for d in range(D):
            for p in range(P):
                tmp = numpy.random.rand(N,N)
                A_data[d,p,:,:] = numpy.dot(tmp.T,tmp)

                if d == 0:
                    A_data[d,p,:,:] += N * numpy.diag(numpy.random.rand(N))

        A = UTPM(A_data)
        l,Q = UTPM.eigh(A)

        L_data = UTPM._diag(l.data)
        L = UTPM(L_data)

        assert_array_almost_equal(UTPM.dot(Q, UTPM.dot(L,Q.T)).data, A.data, decimal = 13)

        lbar = UTPM(numpy.random.rand(*(D,P,N)))
        Qbar = UTPM(numpy.random.rand(*(D,P,N,N)))

        Abar = UTPM.pb_eigh( lbar, Qbar, A, l, Q)

        Abar = Abar.data[0,0]
        Adot = A.data[1,0]

        Lbar = UTPM._diag(lbar.data)[0,0]
        Ldot = UTPM._diag(l.data)[1,0]

        Qbar = Qbar.data[0,0]
        Qdot = Q.data[1,0]

        assert_almost_equal(numpy.trace(numpy.dot(Abar.T, Adot)), numpy.trace( numpy.dot(Lbar.T, Ldot) + numpy.dot(Qbar.T, Qdot)))

    def test_pullback_repeated_eigenvalues(self):
        D,P,N = 2,1,6
        A = UTPM(numpy.zeros((D,P,N,N)))
        V = UTPM(numpy.random.rand(D,P,N,N))
        
        A.data[0,0] = numpy.diag([2,2,3,3.,4,5])
        A.data[1,0] = numpy.diag([5,1,3,1.,1,3])
        
        V,Rtilde = UTPM.qr(V)
        A = UTPM.dot(UTPM.dot(V.T, A), V)

        l,Q = UTPM.eigh(A)

        L_data = UTPM._diag(l.data)
        L = UTPM(L_data)

        assert_array_almost_equal(UTPM.dot(Q, UTPM.dot(L,Q.T)).data, A.data, decimal = 13)

        lbar = UTPM(numpy.random.rand(*(D,P,N)))
        Qbar = UTPM(numpy.random.rand(*(D,P,N,N)))

        Abar = UTPM.pb_eigh( lbar, Qbar, A, l, Q)

        Abar = Abar.data[0,0]
        Adot = A.data[1,0]

        Lbar = UTPM._diag(lbar.data)[0,0]
        Ldot = UTPM._diag(l.data)[1,0]

        Qbar = Qbar.data[0,0]
        Qdot = Q.data[1,0]

        assert_almost_equal(numpy.trace(numpy.dot(Abar.T, Adot)), numpy.trace( numpy.dot(Lbar.T, Ldot) + numpy.dot(Qbar.T, Qdot)))


class TestFunctionOfJacobian(TestCase):
    def test_FtoJT(self):
        (D,P,N) = 2,5,5
        x = UTPM(numpy.random.rand(D,P,N))
        z = x.data[1:,...].reshape((D-1,1,P,N))
        y = x.FtoJT()
        assert_array_equal(y.data.shape, [1,1,5,5])
        assert_array_almost_equal(y.data, z)

    def test_JTtoF(self):
        (D,P,N) = 2,5,5
        x = UTPM(numpy.random.rand(D,P,N))
        y = x.FtoJT()
        z = y.JTtoF()

        assert_array_equal(x.data.shape, z.data.shape)

        assert_array_almost_equal(x.data[1:,...], z.data[:-1,...])


if __name__ == "__main__":
    run_module_suite()
