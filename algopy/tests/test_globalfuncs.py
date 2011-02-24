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


if __name__ == "__main__":
    run_module_suite()



