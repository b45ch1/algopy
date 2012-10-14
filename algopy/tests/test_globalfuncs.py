from numpy.testing import *
import numpy

from algopy.utpm import UTPM
from algopy.globalfuncs import *


class Test_Global_Functions_on_Numpy_instances(TestCase):
    def test_global_unary_function(self):
        x = numpy.ones((6,6))
        assert_array_almost_equal(trace(x), numpy.trace(x))

    def test_global_binary_function(self):
        x = numpy.random.rand(*(3,4))
        y = numpy.random.rand(*(4,3))
        assert_array_almost_equal(dot(x,y),numpy.dot(x,y))

    def test_zeros(self):
        N,M = 3,4
        x = numpy.zeros((1,1),dtype=float)
        y = zeros((N,M), dtype=x)
        assert_array_almost_equal(numpy.zeros((N,M)), y)

    def test_global_zeros_like(self):
        x = numpy.random.rand(*(3,4))
        y = zeros_like(x)
        assert_array_almost_equal(numpy.zeros((3,4)),y)

    def test_global_linalg(self):
        x = numpy.random.rand(5,5)
        assert_array_almost_equal(inv(x),numpy.linalg.inv(x))


class Test_Global_Functions_on_UTPM_instances(TestCase):
    def test_global_unary_function(self):
        D,P,N = 3,4,5
        x = UTPM(numpy.ones((D,P,N,N)))
        assert_array_almost_equal(trace(x).data, N * numpy.ones((D,P)))

    def test_global_binary_function(self):
        D,P,N,M = 3,4,5,6
        x = UTPM(numpy.random.rand(*(D,P,N,M)))
        y = UTPM(numpy.random.rand(*(D,P,M,N)))
        assert_array_almost_equal(dot(x,y).data,UTPM.dot(x,y).data)

    def test_zeros(self):
        D,P,N,M = 3,4,5,6
        x = UTPM(numpy.random.rand(*(D,P)))
        y = zeros((N,M), dtype=x)
        assert_array_almost_equal(numpy.zeros((D,P,N,M)),y.data)


    def test_zeros_with_mpmath_instances_as_dtype(self):
        skiptest = False
        try:
            import mpmath

        except:
            skiptest = True

        if skiptest == False:
            x = UTPM(numpy.array([[mpmath.mpf(3)]]))
            A = zeros((2,2),dtype=x)
            assert_equal( True, isinstance(A.data[0,0,0,0], mpmath.mpf))


    def test_global_zeros_like(self):
        D,P,N,M = 3,4,5,6
        x = UTPM(numpy.random.rand(*(D,P,N,M)))
        y = zeros_like(x)
        assert_array_almost_equal(numpy.zeros((D,P,N,M)),y.data)

    def test_global_linalg(self):
        D,P,N = 3,4,5
        x = UTPM(numpy.random.rand(D,P,N,N))
        assert_array_almost_equal(inv(x).data, UTPM.inv(x).data)

class Test_global_functions(TestCase):


    def test_numpy_overrides(self):

        X = 2 * numpy.random.rand(2,2,2,2)
        Y = 3 * numpy.random.rand(2,2,2,2)

        AX = UTPM(X)
        AY = UTPM(Y)

        assert_array_almost_equal( UTPM.dot(AX,AY).data, dot(AX,AY).data)
        assert_array_almost_equal( UTPM.inv(AX).data,  inv(AX).data)

        assert_array_almost_equal( UTPM.trace(AX).data,  trace(AX).data)


    # def test_convert(self):
        # X1 = 2 * numpy.random.rand(2,2,2,2)
        # X2 = 2 * numpy.random.rand(2,2,2,2)
        # X3 = 2 * numpy.random.rand(2,2,2,2)
        # X4 = 2 * numpy.random.rand(2,2,2,2)
        # AX1 = UTPM(X1)
        # AX2 = UTPM(X2)
        # AX3 = UTPM(X3)
        # AX4 = UTPM(X4)
        # AY = combine_blocks([[AX1,AX2],[AX3,AX4]])

        # assert_array_equal(numpy.shape(AY.data),(2,2,4,4))

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


    def test_expm(self):

        g_data = numpy.array([
                [2954, 141, 17, 16],
                [165, 1110, 5, 2],
                [18, 4, 3163, 374],
                [15, 2, 310, 2411],
                ],dtype=float)

        def transform_params(Y):
            X = exp(Y)
            tsrate, tvrate = X[0], X[1]
            v_unnormalized = zeros(4, dtype=X)
            v_unnormalized[0] = X[2]
            v_unnormalized[1] = X[3]
            v_unnormalized[2] = X[4]
            v_unnormalized[3] = 1.0
            v = v_unnormalized / sum(v_unnormalized)
            return tsrate, tvrate, v

        def eval_f(Y):
            """
            using algopy.expm
            """

            a, b, v = transform_params(Y)

            Q = zeros((4,4), dtype=Y)
            Q[0,0] = 0;    Q[0,1] = a;    Q[0,2] = b;    Q[0,3] = b;
            Q[1,0] = a;    Q[1,1] = 0;    Q[1,2] = b;    Q[1,3] = b;
            Q[2,0] = b;    Q[2,1] = b;    Q[2,2] = 0;    Q[2,3] = a;
            Q[3,0] = b;    Q[3,1] = b;    Q[3,2] = a;    Q[3,3] = 0;

            Q = Q * v
            Q -= diag(sum(Q, axis=1))
            P = expm(Q)
            S = log(dot(diag(v), P))
            return -sum(S * g_data)

        def eval_f_eigh(Y):
            """
            reformulation of eval_f(Y) to use eigh instead of expm
            """
            a, b, v = transform_params(Y)

            Q = zeros((4,4), dtype=Y)
            Q[0,0] = 0;    Q[0,1] = a;    Q[0,2] = b;    Q[0,3] = b;
            Q[1,0] = a;    Q[1,1] = 0;    Q[1,2] = b;    Q[1,3] = b;
            Q[2,0] = b;    Q[2,1] = b;    Q[2,2] = 0;    Q[2,3] = a;
            Q[3,0] = b;    Q[3,1] = b;    Q[3,2] = a;    Q[3,3] = 0;

            Q = dot(Q, diag(v))
            Q -= diag(sum(Q, axis=1))
            va = diag(sqrt(v))
            vb = diag(1./sqrt(v))
            W, U = eigh(dot(dot(va, Q), vb))
            M = dot(U, dot(diag(exp(W)), U.T))
            P = dot(vb, dot(M, va))
            S = log(dot(diag(v), P))
            return -sum(S * g_data)

        def eval_grad_f_eigh(Y):
            """
            compute the gradient of f in the forward mode of AD
            """
            Y = UTPM.init_jacobian(Y)
            retval = eval_f_eigh(Y)
            return UTPM.extract_jacobian(retval)

        def eval_hess_f_eigh(Y):
            """
            compute the hessian of f in the forward mode of AD
            """
            Y = UTPM.init_hessian(Y)
            retval = eval_f_eigh(Y)
            hessian = UTPM.extract_hessian(5, retval)
            return hessian

        def eval_grad_f(Y):
            """
            compute the gradient of f in the forward mode of AD
            """
            Y = UTPM.init_jacobian(Y)
            retval = eval_f(Y)
            return UTPM.extract_jacobian(retval)

        def eval_hess_f(Y):
            """
            compute the hessian of f in the forward mode of AD
            """
            Y = UTPM.init_hessian(Y)
            retval = eval_f(Y)
            hessian = UTPM.extract_hessian(5, retval)
            return hessian

        Y = numpy.zeros(5)
        assert_array_almost_equal(eval_f_eigh(Y), eval_f(Y))
        assert_array_almost_equal(eval_grad_f_eigh(Y), eval_grad_f(Y))
        assert_array_almost_equal(eval_hess_f_eigh(Y), eval_hess_f(Y))




if __name__ == "__main__":
    run_module_suite()



