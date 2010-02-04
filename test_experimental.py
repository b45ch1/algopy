from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utp.utpm import *

class Test_Function_on_UTPM(TestCase):
    
    # def test_lala(self):
        # D,P,N = 1,1,1
        # ax = UTPM(2*numpy.ones((D,P,N)))
        # ay = UTPM(3*numpy.ones((D,P,N)))
        
        # UTPM.__iadd__(ay,ax)
        
        # axbar = UTPM(numpy.zeros((D,P,N)))
        # aybar = UTPM(5*numpy.ones((D,P,N)))
        # print UTPM.pb___iadd__(aybar, ax, ay, ay, out = (axbar, aybar))
        
        # # print ay
        
        
    

        
    def test_reverse_of_chained_getsetitem(self):
        cg = CGraph()
        D,P,N = 1,1,3
        ax = UTPM(numpy.random.rand(D,P,N))
        ay = UTPM(numpy.zeros((D,P,N)))
        fx = Function(ax)
        fy = Function(ay)
     
        for n in range(N):
            fy[n] = fx[n] + fx[n]
            
        cg.independentFunctionList = [fx]
        cg.dependentFunctionList = [fy]

        
        ybar = UTPM(numpy.random.rand(D,P,N))
        cg.pullback([ybar])
        
        assert_array_almost_equal(2*ybar.data, fx.xbar.data)
        
    # def test_reverse_of_chained_iadd(self):
    #     cg = CGraph()
    #     D,P,N = 1,1,1
    #     ax = UTPM(7*numpy.ones((D,P,N)))
    #     ay = UTPM(numpy.zeros((D,P)))
    #     fx = Function(ax)
    #     fy = Function(ay)
        
    #     fy += fx[0]
    #     fy += fx[0]
            
    #     cg.independentFunctionList = [fx]
    #     cg.dependentFunctionList = [fy]
        

        
    #     ybar = UTPM(3*numpy.ones((D,P,N)))
    #     cg.pullback([ybar])
        
        
    #     cg.plot('lala.png')
    #     # assert_array_almost_equal(2*ybar.data, fx.xbar.data)        
        
    # def test_reverse_of_chained_getsetitem2(self):
        # cg = CGraph()
        # D,P,N = 1,1,1
        # ax = UTPM(3*numpy.ones((D,P)))
        # ay = UTPM(numpy.zeros((D,P,N)))
        # fx = Function(ax)
        # fy = Function(ay)
     
        # for n in range(N):
            # fy[n] + fx
            # fy[n] + fx
            
        # cg.independentFunctionList = [fx]
        # cg.dependentFunctionList = [fy]
        
        # ybar = UTPM(5*numpy.ones((D,P,N)))
        # # cg.pullback([ybar])
        
        # print cg
        
        # # assert_array_almost_equal(2*ybar[0].data, fx.xbar.data)
        
    def test_reverse_of_add(self):
        cg = CGraph()
        D,P,N = 1,1,1
        ax = UTPM(3*numpy.ones((D,P)))
        ay = UTPM(numpy.zeros((D,P,N)))
        fx = Function(ax)
        fy = Function(ay)
     
        fy[0] = fy[0] + fx
        fy[0] = fy[0] + fx
        fy[0] = fy[0] + fx
                
        cg.independentFunctionList = [fx]
        cg.dependentFunctionList = [fy]
        
        assert_array_almost_equal(ay[0].data, (3*ax).data)
        ybar = UTPM(5*numpy.ones((D,P,N)))
        cg.pullback([ybar])
        assert_array_almost_equal(3*ybar[0].data, fx.xbar.data)
        assert_array_almost_equal(ay[0].data, (0*ax).data)
       
    
    
    # def test_reverse_of_chained_qr(self):
    #     cg = CGraph()
    #     D,P,N = 1,1,3
    #     ax = UTPM(numpy.random.rand(D,P,N,N))
    #     fx = Function(ax)
        
    #     fQ1,fR1 = Function.qr(fx)
    #     # fQ2,fR2 = Function.qr(fx)
        
    #     fQ = fQ1 #+ fQ2
    #     fR = fR1 #+ fR2
        
    #     fy = Function.dot(fQ,fR)
       
    #     cg.independentFunctionList = [fx]
    #     cg.dependentFunctionList = [fy]
        
        
    #     assert_array_almost_equal(fx.x.data, fy.x.data)
    #     ybar = UTPM(numpy.random.rand(*(D,P,N,N)))

    #     cg.pullback([ybar])
        
    #     print ybar - fx.xbar
    #     # print ax
    #     # print fQ.x
    #     # print fR.x
        
    #     # assert_array_almost_equal(fx, fx.xbar)
        
                
    # def test_reverse_mode_on_linear_function_using_setitem(self):
    #     cg = CGraph()
    #     D,P,N = 1,1,2
    #     ax = UTPM(numpy.random.rand(D,P,N))
    #     ay = UTPM(numpy.zeros((D,P,N)))
    #     aA = UTPM(numpy.random.rand(D,P,N,N))
        
    #     fx = Function(ax)
    #     fA = Function(aA)
    #     fy1 = Function(ay)
        
    #     fy2 = Function.dot(fA,fx)
        
    #     for n in range(N):
    #         fy1[n] = UTPM(numpy.zeros((D,P)))
    #         for k in range(N):
    #             fy1[n] += fA[n,k] * fx[k]
                
    #     cg.independentFunctionList = [fx]
    #     cg.dependentFunctionList = [fy1,fy2]
                
    #     cg.push_forward([UTPM(numpy.random.rand(D,P,N))])
        
    #     # print fy1
    #     # print fy2
            

        
    #     # print cg
        
    #     ybar = UTPM(numpy.zeros((D,P,N)))
    #     ybar.data[0,:,:] = 1.
    #     cg.pullback([ybar, ybar])
        
    #     xbar_correct = UTPM.dot(aA.T, ybar)
        
    #     print xbar_correct - fx.xbar

if __name__ == "__main__":
    run_module_suite() 
