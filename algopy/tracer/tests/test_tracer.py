from numpy.testing import *
from environment import Settings
import os

from algopy.tracer.tracer import *
from algopy.utp.utpm import UTPM
from algopy.utp.utps import UTPS

import numpy

class Test_Function_on_numpy_types(TestCase):
    
    def test_init(self):
        x = 1.
        fx = Function(x)
        assert_array_almost_equal(fx.x,x)
        
    def test_push_forward_add(self):
        x,y = 1.,2.
        fx = Function(x)
        fy = Function(y)
        
        fz = Function.push_forward(numpy.add, [fx,fy])
        assert_almost_equal(fz.x, x + y)
        
        
    def test_push_forward_qr(self):
        x = numpy.random.rand(3,3)
        fx = Function(x)
        fy = Function.push_forward(numpy.linalg.qr, [fx])
        y  = numpy.linalg.qr(x)
        assert_array_almost_equal(fy.x, y)
        
class Test_Function_on_UTPM(TestCase):
    
    def test_init(self):
        D,P = 3,4
        x = UTPM(numpy.ones((D,P)))
        
    def test_push_forward_add(self):
        D,P,N,M = 2,3,4,5
        x = UTPM(numpy.random.rand(D,P,N,M))
        y = UTPM(numpy.random.rand(D,P,N,M))
        fx = Function(x)
        fy = Function(y)
        
        fz = Function.push_forward(UTPM.add, [fx,fy])
        assert_almost_equal(fz.x.data, (x + y).data)
        
        
    def test_pullback_add(self):
        D,P,N,M = 2,3,4,5
        x = UTPM(numpy.random.rand(D,P,N,M))
        y = UTPM(numpy.random.rand(D,P,N,M))
        fx = Function(x)
        fy = Function(y)
        
        fz = Function.push_forward(UTPM.add, [fx,fy])
        fz.xbar = fz.x.zeros_like()
        fx.xbar = fx.x.zeros_like()
        fy.xbar = fy.x.zeros_like()
        
        fz = Function.pullback(fz)
        assert_almost_equal(fx.xbar.data, (fz.xbar * fy.xbar).data)
        assert_almost_equal(fy.xbar.data, (fz.xbar * fx.xbar).data)
        
    def test_get_item(self):
        D,P,N = 2,5,7
        ax = UTPM(numpy.random.rand(D,P,N,N))
        fx = Function(ax)
        
        for r in range(N):
            for c in range(N):
                assert_array_almost_equal( fx[r,c].x.data, ax.data[:,:,r,c])
                
    def test_set_item(self):
        D,P,N = 2,5,7
        ax = UTPM(numpy.zeros((D,P,N)))
        ay = UTPM(numpy.random.rand(*(D,P,N)))
        fx = Function(ax)
        
        for n in range(N):
            fx[n] = 2 * ay[n]
            
        assert_array_almost_equal( fx.x.data, 2*ay.data)
        

class Test_Mixed_Function_Operations(TestCase):
    def test_scalar(self):
        D,P,N = 2,3,4
        x = 4.
        y = 3.
        fx = Function(x)
        
        fz11 = fx + y
        fz12 = fx - y
        fz13 = fx * y
        fz14 = fx / y
        
        fz21 = y + fx
        fz22 = - (y - fx)
        fz23 = y * fx
        fz24 = 1./(y/fx)
        
        assert_array_almost_equal(fz11.x, fz21.x)
        assert_array_almost_equal(fz12.x, fz22.x)
        assert_array_almost_equal(fz13.x, fz23.x)
        assert_array_almost_equal(fz14.x, fz24.x)
        

class Test_CGgraph_on_numpy_operations(TestCase):
    def test_push_forward(self):
        cg = CGraph()
        fx = Function(1.)
        fy = Function(2.)
        
        fz = Function.push_forward(numpy.add, [fx,fy])
        fz = Function.push_forward(numpy.multiply, [fz,fy])

        cg.independentFunctionList = [fx,fy]
        cg.dependentFunctionList = [fz]
        
        x = 32.23
        y = 235.
        cg.push_forward([x,y])
        assert_array_almost_equal( cg.dependentFunctionList[0].x,  (x + y) * y)
        
        
class TestCGraph_on_UTPS(TestCase):
    def test_forward(self):
        cg = CGraph()
        ax = UTPS([3.,1.])
        ay = UTPS([7.,0.])
        fx = Function(ax)
        fy = Function(ay)
        fv1 = fx * fy
        fv2 = (fv1 * fx + fy)*fv1
        cg.independentFunctionList = [fx,fy]
        cg.dependentFunctionList = [fv2]
        cg.push_forward([ax,ay])
        assert_array_almost_equal(cg.dependentFunctionList[0].x.data, ((ax*ay * ax + ay)*ax*ay).data)        
        
        
class Test_CGgraph_on_UTPM(TestCase):
    def test_push_forward(self):
        cg = CGraph()
        D,P,N,M = 2,5,7,11
        aX = UTPM(numpy.random.rand(D,P,N,M))
        aY = UTPM(numpy.random.rand(D,P,N,M))
        fX = Function(aX)
        fY = Function(aY)
        fV1 = fX * fY
        fV2 = (fV1 * fX + fY)*fV1
        cg.independentFunctionList = [fX,fY]
        cg.dependentFunctionList = [fV2]
        cg.push_forward([aX,aY])
        assert_array_almost_equal(cg.dependentFunctionList[0].x.data, ((aX*aY * aX + aY)*aX*aY).data)
        
        
    def test_push_forward_of_qr(self):
        cg = CGraph()
        D,P,N,M = 1,1,3,3
        x = UTPM(numpy.random.rand(D,P,N,M))
        fx = Function(x)
        f = Function.qr(fx)
        
        fQ,fR = f
        
        cg.independentFunctionList = [fx]
        cg.dependentFunctionList = [fQ,fR]
        
        x = UTPM(numpy.random.rand(D,P,N,M))
        cg.push_forward([x])
        Q = cg.dependentFunctionList[0].x
        R = cg.dependentFunctionList[1].x
        
        assert_array_almost_equal(x.data,UTPM.dot(Q,R).data)
        
        
    def test_pullback(self):
        """
        z = x*y*x
          
        dz/dx = 2*x*y
        dz/dy = x*x
        """
        cg = CGraph()
        D,P,N,M = 1,1,1,1
        x = UTPM(numpy.random.rand(D,P,N,M))
        y = UTPM(numpy.random.rand(D,P,N,M))
        fx = Function(x)
        fy = Function(y)
        fv1 = fx * fy * fx
        cg.independentFunctionList = [fx,fy]
        cg.dependentFunctionList = [fv1]
        
        v1bar = UTPM(numpy.ones((D,P,N,M)))
        cg.pullback([v1bar])
        
        # symbolic differentiation
        dzdx = 2.*x*y
        dzdy = x*x
        
        assert_array_almost_equal(dzdx.data, cg.independentFunctionList[0].xbar.data)
        assert_array_almost_equal(dzdy.data, cg.independentFunctionList[1].xbar.data)
        
    def test_pullback2(self):
        """
        z = (x*y*x+y)*x*y
          =  x**3 * y**2 + x * y**2
          
        dz/dx = 3 * x**2 * y**2 + y**2
        dz/dy = 2 * x**3 * y  + 2 * x * y 
        """
        cg = CGraph()
        D,P,N,M = 1,1,1,1
        x = UTPM(numpy.random.rand(D,P,N,M))
        y = UTPM(numpy.random.rand(D,P,N,M))
        fx = Function(x)
        fy = Function(y)
        fv1 = fx * fy
        fv2 = (fv1 * fx + fy)*fv1
        cg.independentFunctionList = [fx,fy]
        cg.dependentFunctionList = [fv2]
        
        v2bar = UTPM(numpy.ones((D,P,N,M)))
        cg.pullback([v2bar])
        
        # symbolic differentiation
        dzdx = 3 * x*x * y*y + y*y
        dzdy = 2 * x*x*x * y  + 2 * x * y
        
        assert_array_almost_equal(dzdx.data, cg.independentFunctionList[0].xbar.data)
        assert_array_almost_equal(dzdy.data, cg.independentFunctionList[1].xbar.data)
        
        
    def test_pullback3(self):
        cg = CGraph()
        D,P,N,M = 2,2,2,2
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

    def test_pullback4(self):
        
        (D,P,M,N) = 2,7,10,3
        A_data = numpy.random.rand(D,P,M,N)

        # make A_data sufficiently regular
        for p in range(P):
            for n in range(N):
                A_data[0,p,n,n] += (N + 1)

        A = UTPM(A_data)
        
        # STEP 1: tracing
        cg = CGraph()
        fA = Function(A)
        fQ,fR = Function.qr(fA)
        cg.independentFunctionList = [fA]
        cg.dependentFunctionList = [fQ,fR]
        
        Q = fQ.x
        R = fR.x

        # STEP 2: pullback

        Qbar_data = numpy.random.rand(*Q.data.shape)
        Rbar_data = numpy.random.rand(*R.data.shape)

        for r in range(N):
            for c in range(N):
                Rbar_data[:,:,r,c] *= (c>r)

        Qbar = UTPM(Qbar_data)
        Rbar = UTPM(Rbar_data)
        
        cg.pullback([Qbar,Rbar])

        Abar = fA.xbar

        for p in range(P):
            Ab = Abar.data[0,p]
            Ad = A.data[1,p]

            Qb = Qbar.data[0,p]
            Qd = Q.data[1,p]

            Rb = Rbar.data[0,p]
            Rd = R.data[1,p]
            assert_almost_equal(numpy.trace(numpy.dot(Ab.T,Ad)), numpy.trace(numpy.dot(Qb.T,Qd) + numpy.dot(Rb.T,Rd)))


    def test_pullback_inv(self):
        """
        test pullback on
        f = inv(A)
        """
        
        (D,P,M,N) = 2,7,10,10
        A_data = numpy.random.rand(D,P,M,N)

        # make A_data sufficiently regular
        for p in range(P):
            for n in range(N):
                A_data[0,p,n,n] += (N + 1)

        A = UTPM(A_data)
        
        # STEP 1: tracing
        cg = CGraph()
        fA = Function(A)
        fAinv = Function.inv(fA)
        cg.independentFunctionList = [fA]
        cg.dependentFunctionList = [fAinv]
        
        Ainv = fAinv.x
        # STEP 2: pullback

        Ainvbar_data = numpy.random.rand(*Ainv.data.shape)

        Ainvbar = UTPM(Ainvbar_data)
        cg.pullback([Ainvbar])
        
        Abar = fA.xbar
        
        for p in range(P):
            Ab = Abar.data[0,p]
            Ad = A.data[1,p]

            Ainvb = Ainvbar.data[0,p]
            Ainvd = Ainv.data[1,p]
            assert_almost_equal(numpy.trace(numpy.dot(Ab.T,Ad)), numpy.trace(numpy.dot(Ainvb.T,Ainvd)))


    def test_pullback_solve(self):
        """
        test pullback on
        f = solve(A,x)
        """
        
        (D,P,M,N,K) = 2,7,10,10,3
        A_data = numpy.random.rand(D,P,M,N)

        # make A_data sufficiently regular
        for p in range(P):
            for n in range(N):
                A_data[0,p,n,n] += (N + 1)

        A = UTPM(A_data)
        x = UTPM(numpy.random.rand(D,P,N,K))
        
        # STEP 1: tracing
        cg = CGraph()
        fA = Function(A)
        fx = Function(x)
        fy = Function.solve(fA,fx)
        cg.independentFunctionList = [fA]
        cg.dependentFunctionList = [fy]
        
        y = fy.x
        # STEP 2: pullback

        ybar_data = numpy.random.rand(*y.data.shape)
        ybar = UTPM(ybar_data)
        cg.pullback([ybar])
        
        Abar = fA.xbar
        xbar = fx.xbar
        
        assert_array_almost_equal(x.data, UTPM.dot(A,y).data)
        
        for p in range(P):
            Ab = Abar.data[0,p]
            Ad = A.data[1,p]
            
            xb = xbar.data[0,p]
            xd = x.data[1,p]
            
            yb = ybar.data[0,p]
            yd = y.data[1,p]
            assert_almost_equal(numpy.trace(numpy.dot(Ab.T,Ad)) + numpy.trace(numpy.dot(xb.T,xd)), numpy.trace(numpy.dot(yb.T,yd)))



    def test_pullback5(self):
        """
        test pullback on::
        
            Q,R = qr(A)
            Rinv = inv(R)
            
        """
        
        (D,P,M,N) = 2,2,3,2
        A_data = numpy.random.rand(D,P,M,N)

        # make A_data sufficiently regular
        for p in range(P):
            for n in range(N):
                A_data[0,p,n,n] += (N + 1)

        A = UTPM(A_data)
        
        # STEP 1: tracing
        cg = CGraph()
        fA = Function(A)
        fQ,fR = Function.qr(fA)
        fRinv = Function.inv(fR)
        cg.independentFunctionList = [fA]
        cg.dependentFunctionList = [fRinv]
        
        Rinv = fRinv.x

        # STEP 2: pullback
        Rinvbar_data = numpy.random.rand(*Rinv.data.shape)

        # make Rinvbar upper triangular
        for r in range(N):
            for c in range(N):
                Rinvbar_data[:,:,r,c] *= (c>=r)

        Rinvbar = UTPM(Rinvbar_data)
        cg.pullback([Rinvbar])

        Abar = fA.xbar

        for p in range(P):
            Ab = Abar.data[0,p]
            Ad = A.data[1,p]

            Rinvb = Rinvbar.data[0,p]
            Rinvd = Rinv.data[1,p]
            
            assert_almost_equal(numpy.trace(numpy.dot(Ab.T,Ad)), numpy.trace(numpy.dot(Rinvb.T,Rinvd)))

        
    def test_reverse_on_basic_element_wise_functions(self):
        cg = CGraph()
        D,P,N,M = 2,5,7,11
        ax = UTPM(numpy.random.rand(D,P,N,M))
        ay = UTPM(numpy.random.rand(D,P,N,M))
        fx = Function(ax)
        fy = Function(ay)
        fv1 = fx * fy
        fv2 = (fv1 * fx + fy)*fv1
        cg.independentFunctionList = [fx,fy]
        cg.dependentFunctionList = [fv2]

        v2bar = UTPM(numpy.zeros((D,P,N,M)))
        v2bar.data[0,:,:,:] = 1.
        cg.pullback([v2bar])

        xbar_reverse = cg.independentFunctionList[0].xbar
        ybar_reverse = cg.independentFunctionList[1].xbar
        
        xbar_symbolic = 3. * ax*ax * ay*ay + ay*ay
        ybar_symbolic = 2.*ax*ax*ax * ay + 2. * ax * ay

        # print xbar_symbolic.tc
        # print xbar_reverse
        # print ybar_symbolic
        # print ybar_reverse
        
        assert_array_almost_equal(xbar_reverse.data, xbar_symbolic.data)
        assert_array_almost_equal(ybar_reverse.data, ybar_symbolic.data)
        
    def test_dot(self):
        cg = CGraph()
        D,P,N,M = 2,5,7,11
        ax = UTPM(numpy.random.rand(D,P,N,M))
        ay = UTPM(numpy.random.rand(D,P,M,N))
        fx = Function(ax)
        fy = Function(ay)
        fz = Function.dot(fx,fy)
        cg.independentFunctionList = [fx,fy]
        cg.dependentFunctionList = [fz]
        
        ax = UTPM(numpy.random.rand(D,P,N,M))
        ay = UTPM(numpy.random.rand(D,P,M,N))
        azbar = UTPM(numpy.random.rand(*fz.x.data.shape))
        cg.push_forward([ax,ay])
        cg.pullback([azbar])
        
        xbar_reverse = cg.independentFunctionList[0].xbar
        ybar_reverse = cg.independentFunctionList[1].xbar        
        
        xbar_symbolic = UTPM.dot(azbar,ay.T)
        ybar_symbolic = UTPM.dot(ax.T,azbar)
        
        assert_array_almost_equal(xbar_reverse.data, xbar_symbolic.data)
        assert_array_almost_equal(ybar_reverse.data, ybar_symbolic.data)
        
        
    def test_transpose(self):
        cg = CGraph()
        D,P,N,M = 2,5,7,11
        ax = UTPM(numpy.random.rand(D,P,N,M))
        fx = Function(ax)
        fy = Function.transpose(fx)
        cg.independentFunctionList = [fx]
        cg.dependentFunctionList = [fy]

        assert_array_equal(fy.shape, (M,N))
        assert_array_equal(fy.x.data.shape, (D,P,M,N))
        
        cg.push_forward([ax])
        assert_array_equal(cg.dependentFunctionList[0].shape, (M,N))
        assert_array_equal(cg.dependentFunctionList[0].x.data.shape, (D,P,M,N))
        
    def test_T(self):
        cg = CGraph()
        D,P,N,M = 2,5,7,11
        ax = UTPM(numpy.random.rand(D,P,N,M))
        fx = Function(ax)
        fy = fx.T
        cg.independentFunctionList = [fx]
        cg.dependentFunctionList = [fy]

        assert_array_equal(fy.shape, (M,N))
        assert_array_equal(fy.x.data.shape, (D,P,M,N))
        
        cg.push_forward([ax])
        assert_array_equal(cg.dependentFunctionList[0].shape, (M,N))
        assert_array_equal(cg.dependentFunctionList[0].x.data.shape, (D,P,M,N))        
    
    def test_part_of_ODOE_objective_function(self):
        D,P,N,M = 2,5,100,3
        MJs = [ UTPM(numpy.random.rand(D,P,N,M)), UTPM(numpy.random.rand(D,P,N,M))]
        cg = CGraph()
        FJs = [Function(MJ) for MJ in MJs]
        FPhi = numpy.sum([ Function.dot(FJ.T, FJ) for FJ in FJs ])
        cg.independentFunctionList = FJs
        cg.dependentFunctionList = [FPhi]
        
        assert_array_equal(FPhi.shape, (M,M))
        cg.push_forward(MJs)
        assert_array_equal(cg.dependentFunctionList[0].x.data.shape, [D,P,M,M])


    def test_reverse_on_getitem_setitem(self):
        cg = CGraph()
        D,P,N,M = 2,3,4,5
        ax = UTPM(numpy.random.rand(D,P,N,M))
        ay = UTPM(numpy.zeros((D,P,N,M)))
        fx = Function(ax)
        fy = Function(ay)
        
        for n in range(N):
            for m in range(M):
                fy[n,m] = fx[n,m]
                
        cg.independentFunctionList = [fx]
        cg.dependentFunctionList = [fy]
        
        assert_array_almost_equal(fx.x.data, fy.x.data)
        
        ybar = UTPM(numpy.zeros((D,P,N,M)))
        ybar.data[0,:,:,:] = 1.
        
        cg.pullback([ybar])
        assert_almost_equal(ybar.data, fx.xbar.data)
        
    def test_reverse_of_chained_dot(self):
        cg = CGraph()
        D,P,N = 1,1,2
        ax = UTPM(numpy.random.rand(D,P,N))
        ay = UTPM(numpy.random.rand(D,P,N))
        fx = Function(ax)
        fy = Function(ay)
        
        fz = Function.dot(fx,fy) + Function.dot(fx,fy)
       
        cg.independentFunctionList = [fx]
        cg.dependentFunctionList = [fz]
                
        cg.push_forward([UTPM(numpy.random.rand(D,P,N))])
       
        zbar = UTPM(numpy.zeros((D,P)))
        zbar.data[0,:] = 1.
        cg.pullback([zbar])

        xbar_correct = 2*ay * zbar
        
        assert_array_almost_equal(xbar_correct.data, fx.xbar.data)

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
        
    def test_reverse_of_chained_iadd(self):
        cg = CGraph()
        D,P,N = 1,1,1
        ax = UTPM(7*numpy.ones((D,P,N)))
        ay = UTPM(numpy.zeros((D,P)))
        fx = Function(ax)
        fy = Function(ay)
        
        fy += fx[0]
        fy += fx[0]
            
        cg.independentFunctionList = [fx]
        cg.dependentFunctionList = [fy]
        
        ybar = UTPM(3*numpy.ones((D,P)))
        cg.pullback([ybar])
        assert_array_almost_equal(2*ybar.data, fx.xbar[0].data)
        assert_array_almost_equal(ybar.data, fy.xbar.data)        
        
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
       
    def test_reverse_of_chained_qr(self):
        cg = CGraph()
        D,P,N = 1,1,3
        ax = UTPM(numpy.random.rand(D,P,N,N))
        fx = Function(ax)
        
        fQ1,fR1 = Function.qr(fx)
        fQ2,fR2 = Function.qr(fx)
        
        fQ = fQ1  + fQ2
        fR = fR1  + fR2
        
        fy = Function.dot(fQ,fR)
       
        cg.independentFunctionList = [fx]
        cg.dependentFunctionList = [fy]
        
        
        assert_array_almost_equal(4*fx.x.data, fy.x.data)
        ybar = UTPM(numpy.random.rand(*(D,P,N,N)))

        cg.pullback([ybar])
        
        assert_array_almost_equal((4*ybar).data,fx.xbar.data)
                
    def test_reverse_mode_on_linear_function_using_setitem(self):
        cg = CGraph()
        D,P,N = 1,1,2
        ax = UTPM(numpy.random.rand(D,P,N))
        ay = UTPM(numpy.zeros((D,P,N)))
        aA = UTPM(numpy.random.rand(D,P,N,N))
        
        fx = Function(ax)
        fA = Function(aA)
        fy1 = Function(ay)

        for n in range(N):
            fy1[n] = UTPM(numpy.zeros((D,P)))
            for k in range(N):
                fy1[n] += fA[n,k] * fx[k]
                
        cg.independentFunctionList = [fx]
        cg.dependentFunctionList = [fy1]
                
        cg.push_forward([UTPM(numpy.random.rand(D,P,N))])
        ybar = UTPM(numpy.zeros((D,P,N)))
        ybar.data[0,:,:] = 1.
        cg.pullback([ybar])
        
        xbar_correct = UTPM.dot(aA.T, ybar)
        
        assert_array_almost_equal(xbar_correct.data, fx.xbar.data)



class Test_CGraph_Plotting(TestCase):
    def test_simple(self):
        cg = CGraph()
        D,P,N,M = 2,5,7,11
        ax = UTPM(numpy.random.rand(D,P,N,M))
        ay = UTPM(numpy.random.rand(D,P,M,N))
        fx = Function(ax)
        fy = Function(ay)
        fz = Function.dot(fx,fy)
        for i in range(10):
            fz = Function.dot(fz,fy.T)
            fz = 3 * Function.dot((fz * fx + fx),fy)
            for n in range(N):
                fz[0,0] += fz[n,0]

        cg.independentFunctionList = [fx,fy]
        cg.dependentFunctionList = [fz]
        cg.plot(os.path.join(Settings.output_dir,'test_simple.svg'))
        


if __name__ == "__main__":
    run_module_suite()
