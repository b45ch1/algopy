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



if __name__ == "__main__":
    run_module_suite()



