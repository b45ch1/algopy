"""
check that algopy.linalg.qr(x) correctly calls either

* numpy.linalg.qr(x)
* UTPM.linalg.qr(x)
* or Function.qr(x)

depending on the type of x.
"""



from numpy.testing import *
import numpy

from algopy import UTPM, Function, CGraph, diag, sum
from algopy.linalg import *

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



