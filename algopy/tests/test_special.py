from numpy.testing import *
import numpy

from algopy import UTPM, Function
from algopy.special import *

class Test_ScipySpecialFunctions(TestCase):

    def test_hyp1f1(self):
        """
        check that algopy.special.hyp1f1 can be called with
        UTPM and Function instances as arguments
        """

        a, b, x = 1., 2., 3.
        y1 = hyp1f1(a, b, x)

        a, b, x = 1., 2., UTPM(3.* numpy.ones((1,1)))
        y2 = hyp1f1(a, b, x)
        assert_almost_equal(y1, y2.data[0,0])

        a, b, x = 1., 2., Function(3.)
        y3 = hyp1f1(a, b, x)
        assert_almost_equal(y1, y3.x)

    def test_hyp0f1(self):
        """
        check that algopy.special.hyp0f1 can be called with
        UTPM and Function instances as arguments
        """

        b, x = 2., 3.
        y1 = hyp0f1(b, x)

        b, x = 2., UTPM(3.* numpy.ones((1,1)))
        y2 = hyp0f1(b, x)
        assert_almost_equal(y1, y2.data[0,0])

        b, x = 2., Function(3.)
        y3 = hyp0f1(b, x)
        assert_almost_equal(y1, y3.x)

    def test_erf(self):
        """
        check that algopy.special.erf can be called with
        UTPM and Function instances as arguments
        """

        x = 3.
        y1 = erf(x)

        x = UTPM(3.* numpy.ones((1,1)))
        y2 = erf(x)
        assert_almost_equal(y1, y2.data[0,0])

        x = Function(3.)
        y3 = erf(x)
        assert_almost_equal(y1, y3.x)

    def test_erfi(self):
        """
        check that algopy.special.erfi can be called with
        UTPM and Function instances as arguments
        """

        x = 3.
        y1 = erfi(x)

        x = UTPM(3.* numpy.ones((1,1)))
        y2 = erfi(x)
        assert_almost_equal(y1, y2.data[0,0])

        x = Function(3.)
        y3 = erfi(x)
        assert_almost_equal(y1, y3.x)





if __name__ == "__main__":
    run_module_suite()



