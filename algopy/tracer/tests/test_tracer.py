from numpy.testing import *
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
        
        fz = Function.push_forward(numpy.add, (fx,fy))
        assert_almost_equal(fz.x, x + y)
        
        
    def test_push_forward_qr(self):
        x = numpy.random.rand(3,3)
        fx = Function(x)
        fy = Function.push_forward(numpy.linalg.qr, fx)
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
        
        fz = Function.push_forward(UTPM.add, (fx,fy))
        assert_almost_equal(fz.x.data, (x + y).data)
        
        
    def test_pullback_add(self):
        D,P,N,M = 2,3,4,5
        x = UTPM(numpy.random.rand(D,P,N,M))
        y = UTPM(numpy.random.rand(D,P,N,M))
        fx = Function(x)
        fy = Function(y)
        
        fz = Function.push_forward(UTPM.add, (fx,fy))
        fz.xbar = fz.x.zeros_like()
        fx.xbar = fx.x.zeros_like()
        fy.xbar = fy.x.zeros_like()
        
        fz = Function.pullback(fz)
        assert_almost_equal(fx.xbar.data, (fz.xbar * fy.xbar).data)
        assert_almost_equal(fy.xbar.data, (fz.xbar * fx.xbar).data)
        

class Test_CGgraph_on_numpy_operations(TestCase):
    def test_push_forward(self):
        cg = CGraph()
        fx = Function(1.)
        fy = Function(2.)
        
        fz = Function.push_forward(numpy.add, (fx,fy))
        fz = Function.push_forward(numpy.multiply, (fz,fy))

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

        
        
        

if __name__ == "__main__":
    run_module_suite()
