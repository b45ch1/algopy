from numpy.testing import *
import numpy
import numpy.random

from algopy.utp.utpm import UTPM
from algopy.utp.utps import UTPS
from algopy.utp.utils import *

class TestUtils ( TestCase ):
    def test_utpm2utps(self):
        x = UTPM( numpy.zeros((2,5,3,7),dtype=float))
        y = utpm2utps(x)
        
        assert_array_equal(y.shape, (3,7))
        
    def test_utps2utpm(self):
        D,P,N,M = 2,5,3,7
        x = numpy.array([[UTPS(numpy.random.rand(D,P)) for m in range(M)] for n in range(N)])
        y = utps2utpm(x)
        
        assert_array_equal(y.tc.shape, (D,P,N,M))


if __name__ == "__main__":
    run_module_suite()
