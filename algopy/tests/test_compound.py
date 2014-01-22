from numpy.testing import *
import numpy

from algopy import UTPM, Function, CGraph, sum, zeros, diag, dot, qr
from algopy.linalg.compound import expm

class Test_NumpyScipyLinalgFunctions(TestCase):


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



