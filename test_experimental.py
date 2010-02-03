from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utp.utpm import *

class Test_Function_on_UTPM(TestCase):
                
        
    def test_pullback3(self):
        cg = CGraph()
        D,P,N,M = 1,1,2,2
        x = UTPM(numpy.random.rand(*(D,P,N,M)))
            
        fx = Function(x)
        f = Function.qr(fx)
        
        fQ,fR = f
        
        cg.independentFunctionList = [fx]
        cg.dependentFunctionList = [fQ,fR]
        
        Qbar = UTPM(numpy.ones((D,P,N,M)))
        Rbar = UTPM(numpy.ones((D,P,N,M)))
        
        # print cg
        cg.pullback([Qbar,Rbar])
        # print cg
        

if __name__ == "__main__":
    run_module_suite() 
