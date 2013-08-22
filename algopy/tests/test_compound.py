from numpy.testing import *
import numpy

from algopy import UTPM, Function, CGraph, sum, zeros, diag, dot, qr
from algopy.linalg.compound import svd, expm

class Test_NumpyScipyLinalgFunctions(TestCase):

    def test_svd(self):
        D,P,M,N = 3,1,5,2
        A = UTPM(numpy.random.random((D,P,M,N)))

        U,s,V = svd(A)

        S = zeros((M,N),dtype=A)
        S[:N,:N] = diag(s)

        assert_array_almost_equal( (dot(dot(U, S), V.T) - A).data, 0.)
        assert_array_almost_equal( (dot(U.T, U) - numpy.eye(M)).data, 0.)
        assert_array_almost_equal( (dot(U, U.T) - numpy.eye(M)).data, 0.)
        assert_array_almost_equal( (dot(V.T, V) - numpy.eye(N)).data, 0.)
        assert_array_almost_equal( (dot(V, V.T) - numpy.eye(N)).data, 0.)

    def test_svd1(self):
        D,P,M,N = 2,1,2,2

        U = UTPM(numpy.random.random((D,P,M,M)))
        S = UTPM(numpy.zeros((D,P,M,N)))
        V = UTPM(numpy.random.random((D,P,N,N)))

        U = qr(U)[0]
        V = qr(V)[0]

        S.data[1,0, 0 ,0] = 1.
        S.data[1,0, 1, 1] = 1.

        A = dot(U, dot(S,V))

        U2,s2,V2 = svd(A)
        S2 = zeros((M,N),dtype=A)
        S2[:N,:N] = diag(s2)

        A2 = dot(dot(U2, S2), V2.T)

        # print 'S=', S
        # print 'S2=', S2

        # print A - A2


        assert_array_almost_equal( (A2 - A).data, 0.)
        assert_array_almost_equal( (dot(U2.T, U2) - numpy.eye(M)).data, 0.)
        assert_array_almost_equal( (dot(U2, U2.T) - numpy.eye(M)).data, 0.)
        assert_array_almost_equal( (dot(V2.T, V2) - numpy.eye(N)).data, 0.)
        assert_array_almost_equal( (dot(V2, V2.T) - numpy.eye(N)).data, 0.)

    def test_svd2(self):
        D,P,M,N = 2,1,3,3

        U = UTPM(numpy.random.random((D,P,M,M)))
        S = UTPM(numpy.zeros((D,P,M,N)))
        V = UTPM(numpy.random.random((D,P,N,N)))

        U = qr(U)[0]
        V = qr(V)[0]

        S.data[1,0, 0 ,0] = 1.
        S.data[1,0, 1, 1] = 1.

        A = dot(U, dot(S,V))

        U2,s2,V2 = svd(A)
        S2 = zeros((M,N),dtype=A)
        S2[:N,:N] = diag(s2)

        A2 = dot(dot(U2, S2), V2.T)

        # print 'S=', S
        # print 'S2=', S2

        # print A - A2
        # print 'U2=\n', U2
        # print 'V2=\n', V2
        # print 'dot(U2.T, U2)=\n',dot(U2.T, U2)
        # print 'dot(V2.T, V2)=\n',dot(V2.T, V2)


        assert_array_almost_equal( (A2 - A).data, 0.)
        assert_array_almost_equal( (dot(U2.T, U2) - numpy.eye(M)).data, 0.)
        assert_array_almost_equal( (dot(U2, U2.T) - numpy.eye(M)).data, 0.)
        assert_array_almost_equal( (dot(V2.T, V2) - numpy.eye(N)).data, 0.)
        assert_array_almost_equal( (dot(V2, V2.T) - numpy.eye(N)).data, 0.)

    def test_svd3(self):
        """
        M == N, repeated singular values
        """
        D,P,M,N = 4,1,4,4

        U = UTPM(numpy.random.random((D,P,M,M)))
        S = UTPM(numpy.zeros((D,P,M,N)))
        V = UTPM(numpy.random.random((D,P,N,N)))

        U = qr(U)[0]
        V = qr(V)[0]

        # zeroth coefficient
        S.data[0,0, 0 ,0] = 1.
        S.data[0,0, 1, 1] = 1.
        S.data[0,0, 2, 2] = 0
        S.data[0,0, 3, 3] = 0

        # first coefficient
        S.data[1,0, 0 ,0] = 1.
        S.data[1,0, 1, 1] = -2.
        S.data[1,0, 2, 2] = 0
        S.data[1,0, 3, 3] = 0


        A = dot(U, dot(S,V))

        U2,s2,V2 = svd(A)
        S2 = zeros((M,N),dtype=A)
        S2[:N,:N] = diag(s2)

        A2 = dot(dot(U2, S2), V2.T)

        # print 'S=', S
        # print 'S2=', S2

        # print A - A2
        # print 'U2=\n', U2
        # print 'V2=\n', V2
        # print 'dot(U2.T, U2)=\n',dot(U2.T, U2)
        # print 'dot(V2.T, V2)=\n',dot(V2.T, V2)


        assert_array_almost_equal( (A2 - A).data, 0.)
        assert_array_almost_equal( (dot(U2.T, U2) - numpy.eye(M)).data, 0.)
        assert_array_almost_equal( (dot(U2, U2.T) - numpy.eye(M)).data, 0.)
        assert_array_almost_equal( (dot(V2.T, V2) - numpy.eye(N)).data, 0.)
        assert_array_almost_equal( (dot(V2, V2.T) - numpy.eye(N)).data, 0.)


    def test_svd4(self):
        """
        M > N
        """
        D,P,M,N = 4,1,5,3

        U = UTPM(numpy.random.random((D,P,M,M)))
        S = UTPM(numpy.zeros((D,P,M,N)))
        V = UTPM(numpy.random.random((D,P,N,N)))

        U = qr(U)[0]
        V = qr(V)[0]

        # zeroth coefficient
        S.data[0,0, 0 ,0] = 0.
        S.data[0,0, 1, 1] = 0.
        S.data[0,0, 2, 2] = 1.

        # first coefficient
        S.data[1,0, 0 ,0] = 0.
        S.data[1,0, 1, 1] = 0.

        A = dot(U, dot(S,V))

        U2,s2,V2 = svd(A)
        S2 = zeros((M,N),dtype=A)
        S2[:N,:N] = diag(s2)

        A2 = dot(dot(U2, S2), V2.T)

        # print 'S=', S
        # print 'S2=', S2

        # print A - A2
        # print 'U2=\n', U2
        # print 'V2=\n', V2
        # print 'dot(U2.T, U2)=\n',dot(U2.T, U2)
        # print 'dot(V2.T, V2)=\n',dot(V2.T, V2)

        assert_array_almost_equal( (A2 - A).data, 0.)
        assert_array_almost_equal( (dot(U2.T, U2) - numpy.eye(M)).data, 0.)
        assert_array_almost_equal( (dot(U2, U2.T) - numpy.eye(M)).data, 0.)
        assert_array_almost_equal( (dot(V2.T, V2) - numpy.eye(N)).data, 0.)
        assert_array_almost_equal( (dot(V2, V2.T) - numpy.eye(N)).data, 0.)


    def test_svd5(self):
        """
        M < N
        """
        D,P,M,N = 4,1,3,5
        K = min(M,N)

        U = UTPM(numpy.random.random((D,P,M,M)))
        S = UTPM(numpy.zeros((D,P,M,N)))
        V = UTPM(numpy.random.random((D,P,N,N)))

        U = qr(U)[0]
        V = qr(V)[0]

        # zeroth coefficient
        S.data[0,0, 0 ,0] = 1.
        S.data[0,0, 1, 1] = 1.
        S.data[0,0, 2, 2] = 0.

        # first coefficient
        S.data[1,0, 0 ,0] = 0.
        S.data[1,0, 1, 1] = 0.

        A = dot(U, dot(S,V))
        U2,s2,V2 = svd(A)
        S2 = zeros((M,N),dtype=A)
        S2[:K,:K] = diag(s2)

        A2 = dot(dot(U2, S2), V2.T)

        # print 'S=', S
        # print 'S2=', S2

        # print A - A2
        # print 'U2=\n', U2
        # print 'V2=\n', V2
        # print 'dot(U2.T, U2)=\n',dot(U2.T, U2)
        # print 'dot(V2.T, V2)=\n',dot(V2.T, V2)

        assert_array_almost_equal( (A2 - A).data, 0.)
        assert_array_almost_equal( (dot(U2.T, U2) - numpy.eye(M)).data, 0.)
        assert_array_almost_equal( (dot(U2, U2.T) - numpy.eye(M)).data, 0.)
        assert_array_almost_equal( (dot(V2.T, V2) - numpy.eye(N)).data, 0.)
        assert_array_almost_equal( (dot(V2, V2.T) - numpy.eye(N)).data, 0.)

    def test_svd_example2(self):
        """
        Example 2 from the paper "Numerical Computatoin of an Analytic Singular
        Value Decomposition of a Matrix Valued Function", by Bunse-Gerstner, Byers,
        Mehrmann, Nichols
        """

        D,P,M,N = 2,1,2,2

        U = UTPM(numpy.zeros((D,P,M,N)))
        S = UTPM(numpy.zeros((D,P,M,N)))
        V = UTPM(numpy.zeros((D,P,M,N)))

        U.data[0,0, ...] = numpy.eye(2)
        V.data[0,0, ...] = numpy.eye(2)
        S.data[0,0, ...] = numpy.eye(2)
        S.data[1,0, 0 ,0] = - 1.
        S.data[1,0, 1, 1] = 1.

        A = dot(U, dot(S, V.T))

        U2,s2,V2 = svd(A)

        S2 = zeros((M,N),dtype=A)
        S2[:N,:N] = diag(s2)

        assert_array_almost_equal( (dot(dot(U2, S2), V2.T) - A).data, 0.)
        assert_array_almost_equal( (dot(U2.T, U2) - numpy.eye(M)).data, 0.)
        assert_array_almost_equal( (dot(U2, U2.T) - numpy.eye(M)).data, 0.)
        assert_array_almost_equal( (dot(V2.T, V2) - numpy.eye(N)).data, 0.)
        assert_array_almost_equal( (dot(V2, V2.T) - numpy.eye(N)).data, 0.)

    def test_expm(self):

        def f(x):
            x = x.reshape((2,2))
            return sum(expm(x))

        x = numpy.random.random(2*2)


        # forward mode

        ax = UTPM.init_jacobian(x)
        ay = f(ax)
        g1  = UTPM.extract_jacobian(ay)

        # reverse mode

        cg = CGraph()
        ax = Function(x)
        ay = f(ax)
        cg.independentFunctionList = [ax]
        cg.dependentFunctionList = [ay]

        g2 = cg.gradient(x)

        assert_array_almost_equal(g1, g2)

if __name__ == "__main__":
    run_module_suite()



