from numpy.testing import *
import numpy

from algopy.utp.utpm import UTPM
from algopy.globalfuncs import *


class Test_Global_Functions_on_Numpy_instances(TestCase):
    def test_global_unary_function(self):
        x = numpy.ones((6,6))
        assert_array_almost_equal(trace(x), numpy.trace(x))
        
    def test_global_binary_function(self):
        x = numpy.random.rand(*(3,4))
        y = numpy.random.rand(*(4,3))
        assert_array_almost_equal(dot(x,y),numpy.dot(x,y))
        
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
        
    def test_global_linalg(self):
        D,P,N = 3,4,5
        x = UTPM(numpy.random.rand(D,P,N,N))
        assert_array_almost_equal(inv(x).data, UTPM.inv(x).data)

if __name__ == "__main__":
    run_module_suite()



