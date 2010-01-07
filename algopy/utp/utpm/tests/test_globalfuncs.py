from numpy.testing import *
import numpy

from algopy.utp.utpm import *
from algopy.utp.utpm.globalfuncs import *

class Test_global_functions(TestCase):
  
 
    def test_numpy_overrides(self):
        """
        this checks _only_ if calling the operations is ok
        """
        X = 2 * numpy.random.rand(2,2,2,2)
        Y = 3 * numpy.random.rand(2,2,2,2)

        AX = UTPM(X)
        AY = UTPM(Y)

        assert_array_almost_equal( UTPM.dot(AX,AY).data, dot(AX,AY).data)
        assert_array_almost_equal( UTPM.inv(AX).data,  inv(AX).data)
        assert_array_almost_equal( AX.trace().data,  trace(AX).data)

    
    def test_convert(self):
        X1 = 2 * numpy.random.rand(2,2,2,2)
        X2 = 2 * numpy.random.rand(2,2,2,2)
        X3 = 2 * numpy.random.rand(2,2,2,2)
        X4 = 2 * numpy.random.rand(2,2,2,2)
        AX1 = UTPM(X1)
        AX2 = UTPM(X2)
        AX3 = UTPM(X3)
        AX4 = UTPM(X4)
        AY = combine_blocks([[AX1,AX2],[AX3,AX4]])

        assert_array_equal(numpy.shape(AY.data),(2,2,4,4))

if __name__ == "__main__":
    run_module_suite()
 
