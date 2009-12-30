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





if __name__ == "__main__":
    run_module_suite()