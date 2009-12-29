from numpy.testing import *
import numpy

from algopy.utp.utpm import *

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


    def test__diag(self):
        D,P,N = 2,3,4
        x = numpy.random.rand(D,P,N)

        X = UTPM._diag(x)

        for n in range(N):
            assert_almost_equal( x[...,n], X[...,n,n])

    def test__transpose(self):
        D,P,M,N = 2,3,4,5
        X_data = numpy.random.rand(D,P,N,M)
        Y_data = UTPM._transpose(X_data)

        assert_array_almost_equal(X_data.transpose((0,1,3,2)), Y_data) 


class Test_Linear_Algebra_Pullback(TestCase):
    def test__eig_pullback(self):
        (D,P,N) = 2,1,3
        A_data = numpy.zeros((D,P,N,N))
        for d in range(D):
            for p in range(P):
                tmp = numpy.random.rand(N,N)
                A_data[d,p,:,:] = numpy.dot(tmp.T,tmp)

                if d == 0:
                    A_data[d,p,:,:] += N * numpy.diag(numpy.random.rand(N))

        A = UTPM(A_data)

        l,Q = UTPM.eig(A)

        L_data = UTPM._diag(l.data)

        L = UTPM(L_data)

        assert_array_almost_equal(UTPM.dot(Q, UTPM.dot(L,Q.T)).data, A.data, decimal = 13)

        lbar_data = numpy.random.rand(*(D,P,N))
        Qbar_data = numpy.random.rand(*(D,P,N,N))
        Abar_data = numpy.zeros((D,P,N,N))

        UTPM._eig_pullback( Qbar_data, lbar_data, A.data, Q.data, l.data, out = Abar_data)

        Abar = Abar_data[0,0]
        Adot = A.data[1,0]

        Lbar = UTPM._diag(lbar_data)[0,0]
        Ldot = UTPM._diag(l.data)[1,0]

        Qbar = Qbar_data[0,0]
        Qdot = Q.data[1,0]

        assert_almost_equal(numpy.trace(numpy.dot(Abar.T, Adot)), numpy.trace( numpy.dot(Lbar.T, Ldot) + numpy.dot(Qbar.T, Qdot)))


    def test__qr_pullback(self):
        (D,P,M,N) = 2,1,2,2

        A_data = numpy.random.rand(D,P,M,N)
        # make A_data sufficiently regular
        for p in range(P):
            for n in range(N):
                A_data[0,p,n,n] += (N + 1)

        A = UTPM(A_data)

        Q,R = UTPM.qr(A)

        assert_array_equal( Q.data.shape, [D,P,M,N])
        assert_array_equal( R.data.shape, [D,P,N,N])
        assert_array_almost_equal( (UTPM.dot(Q,R)).data, A.data, decimal = 14)

        Qbar_data = numpy.random.rand(*Q.data.shape)
        Rbar_data = numpy.random.rand(*R.data.shape)
        Abar_data = numpy.zeros(A.data.shape)


        UTPM._qr_pullback(Qbar_data, Rbar_data, A.data, Q.data, R.data, out = Abar_data )


        Abar = Abar_data[0,0]
        Adot = A_data[1,0]

        Qbar = Qbar_data[0,0]
        Qdot = Q.data[1,0]

        Rbar = Rbar_data[0,0]
        Rdot = R.data[1,0]

        assert_almost_equal(numpy.trace(numpy.dot(Abar.T,Adot)), numpy.trace(numpy.dot(Qbar.T,Qdot) + numpy.dot(Rbar.T,Rdot)))

    def test__qr_pullback_rectangular_A(self):
        (D,P,M,N) = 2,1,5,2

        A_data = numpy.random.rand(D,P,M,N)
        # make A_data sufficiently regular
        for p in range(P):
            for n in range(N):
                A_data[0,p,n,n] += (N + 1)

        A = UTPM(A_data)

        Q,R = UTPM.qr(A)

        assert_array_equal( Q.data.shape, [D,P,M,N])
        assert_array_equal( R.data.shape, [D,P,N,N])
        assert_array_almost_equal( (UTPM.dot(Q,R)).data, A.data, decimal = 14)

        Qbar_data = numpy.random.rand(*Q.data.shape)
        Rbar_data = numpy.random.rand(*R.data.shape)
        Abar_data = numpy.zeros(A.data.shape)


        UTPM._qr_pullback(Qbar_data, Rbar_data, A.data, Q.data, R.data, out = Abar_data )


        Abar = Abar_data[0,0]
        Adot = A_data[1,0]

        Qbar = Qbar_data[0,0]
        Qdot = Q.data[1,0]

        Rbar = Rbar_data[0,0]
        Rdot = R.data[1,0]

        assert_almost_equal(numpy.trace(numpy.dot(Abar.T,Adot)), numpy.trace(numpy.dot(Qbar.T,Qdot) + numpy.dot(Rbar.T,Rdot)))
        



if __name__ == "__main__":
    run_module_suite()