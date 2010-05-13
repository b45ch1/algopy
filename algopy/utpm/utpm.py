"""
Implementation of the univariate matrix polynomial.
The algebraic class is

M[t]/<t^D>

where M is the ring of matrices and t an external parameter

"""

import numpy.linalg
import numpy

from ..base_type import Ring
from algorithms import RawAlgorithmsMixIn

class UTPM(Ring, RawAlgorithmsMixIn):
    """
    UTPM == Univariate Taylor Polynomial of Matrices
    This class implements univariate Taylor arithmetic on matrices, i.e.
    [A] = \sum_{d=0}^D A_d t^d
    A_d = \frac{d^d}{dt^d}|_{t=0} \sum_{c=0}^D A_c t^c

    in vector forward mode
    Input:
    in the most general form, the input is a 4-tensor.
    We use the notation:
    D: degree of the Taylor series
    P: number of directions
    N: number of rows of A_0
    M: number of cols of A_0

    shape([A]) = (D,P,N,M)
    The reason for this choice is that the (N,M) matrix is the elementary type, so that memory should be contiguous. Then, at each operation, the code performed to compute
    v_d has to be repeated for every direction.
    E.g. a multiplication
    [w] = [u]*[v] =
    [[u_11, ..., u_1Ndir],
    ...
    [u_D1, ..., u_DNdir]]  +
    [[v11, ..., v_1Ndir],
    ...
    [v_D1, ..., v_DNdir]] =
    [[ u_11 + v_11, ..., u_1Ndir + v_1Ndir],
    ...
    [[ u_D1 + v_D1, ..., u_DNdir + v_DNdir]]

    For ufuncs this arrangement is advantageous, because in this order, memory chunks of size Ndir are used and the operation on each element is the same. This is desireable to avoid cache misses.
    See for example __mul__: there, operations of self.data[:d+1,:,:,:]* rhs.data[d::-1,:,:,:] has to be performed. One can see, that contiguous memory blocks are used for such operations.

    A disadvantage of this arrangement is: it seems unnatural. It is easier to regard each direction separately.
    """
    
    __array_priority__ = 2
    
    def __init__(self, X, Xdot = None):
        """ INPUT:	shape([X]) = (D,P,N,M)
        """
        Ndim = numpy.ndim(X)
        if Ndim >= 2:
            self.data = numpy.asarray(X)
            self.data = self.data
        else:
            raise NotImplementedError
            
    def __getitem__(self, sl):
        if type(sl) == int or sl == Ellipsis or isinstance(sl, slice):
            sl = (sl,)
        
        tmp = self.data.__getitem__((slice(None),slice(None)) + tuple(sl))
        return self.__class__(tmp)
        
    def __setitem__(self, sl, rhs):
        if isinstance(rhs, UTPM):
            if type(sl) == int or sl == Ellipsis or isinstance(sl, slice):
                sl = (sl,)
            return self.data.__setitem__((slice(None),slice(None)) + sl, rhs.data)
        else:
            if type(sl) == int or sl == Ellipsis or isinstance(sl, slice):
                sl = (sl,)
            self.data.__setitem__((slice(1,None),slice(None)) + sl, 0)
            return self.data.__setitem__((0,slice(None)) + sl, rhs)
        
    @classmethod
    def pb___getitem__(cls, ybar, x, sl, y, out = None):
        """
        y = getitem(x, sl)
        
        Warning:
        this includes a workaround for tuples, e.g. for Q,R = qr(A)
        where A,Q,R are Function objects
        """
        if out == None:
            raise NotImplementedError('I\'m not sure if this makes sense')
            
        if isinstance( out[0], tuple):
            tmp = list(out[0])
            tmp[sl] += ybar
            
        else:
            out[0][sl] = ybar
            
        return out
        
    @classmethod
    def pb___setitem__(cls, y, sl, x, out = None):
        """
        y.__setitem(sl,x)
        """
        
        if out == None:
            raise NotImplementedError('I\'m not sure if this makes sense')
        
        ybar, dummy, xbar = out
        # print 'xbar =', xbar
        # print 'ybar =', ybar
        xbar += ybar[sl]
        ybar[sl].data[...] = 0.
        # print 'funcargs=',funcargs
        # print y[funcargs[0]]
        
        
    @classmethod
    def broadcast(cls, x,y):
        """ supposed to work like numpy.broadcast but for UTPM instances
        FIXME: returns the shape at the moment! 
        """
        D,P = x.data.shape[:2]
        class Dummy:
            shape = (D,P) + numpy.broadcast(x.data[0,0,...], y.data[0,0,...]).shape

        return Dummy()
        
    @classmethod
    def postpend_ones(cls, x, y):
        """
        a helper function for broadcasting. The problem is that numpy broadcasting
        prepends ones if necessary, however, the memory layout of UTPM.data
        requires to postpend ones.
        
        This function postpends these ones and returns a new UTPM instance without
        copying memory or the like.
        
        Example:
        
        x.data.shape = (2,3,4,5)
        y.data.shape = (2,3)
        
        then postpend_ones returns two UTPM instances u,v with
        u.data.shape = (2,3,4,5)
        v.data.shape = (2,3,1,1)
        """
        return cls(x.data.reshape(x.data.shape + tuple([1]*(y.data.ndim - x.data.ndim)))),\
                cls(y.data.reshape(y.data.shape + tuple([1]*(x.data.ndim - y.data.ndim))))
        

    
    def __add__(self,rhs):
        if numpy.isscalar(rhs) or isinstance(rhs,numpy.ndarray):
            retval = UTPM(numpy.copy(self.data))
            retval.data[0,:] += rhs
            return retval
        else:
            return UTPM(self.data + rhs.data)

    def __sub__(self,rhs):
        if numpy.isscalar(rhs) or isinstance(rhs,numpy.ndarray):
            retval = UTPM(numpy.copy(self.data))
            retval.data[0,:] -= rhs
            return retval
        else:
            return UTPM(self.data - rhs.data)

    def __mul__(self,rhs):
        retval = self.clone()
        retval.__imul__(rhs)
        return retval

    def __div__(self,rhs):
        retval = self.clone()
        retval.__idiv__(rhs)
        return retval

    def __radd__(self,rhs):
        return self + rhs

    def __rsub__(self, other):
        return -self + other

    def __rmul__(self,rhs):
        return self * rhs

    def __rdiv__(self, rhs):
        tmp = self.zeros_like()
        tmp.data[0,:,:,:] = rhs
        return tmp/self
        
    def __iadd__(self,rhs):
        if numpy.isscalar(rhs) or isinstance(rhs,numpy.ndarray):
            self.data[0,...] += rhs
        else:
            self.data[...] += rhs.data[...]
        return self
        
    def __isub__(self,rhs):
        if numpy.isscalar(rhs) or isinstance(rhs,numpy.ndarray):
            self.data[0,...] -= rhs
        else:
            self.data[...] -= rhs.data[...]
        return self
        
    def __imul__(self,rhs):
        (D,P) = self.data.shape[:2]
        if numpy.isscalar(rhs) or isinstance(rhs,numpy.ndarray):
            for d in range(D):
                for p in range(P):
                    self.data[d,p,...] *= rhs
        else:
            for d in range(D)[::-1]:
                for p in range(P):
                    self.data[d,p,...] *= rhs.data[0,p,...]
                    for c in range(d):
                        self.data[d,p,...] += self.data[c,p,...] * rhs.data[d-c,p,...]
        return self
        
    def __idiv__(self,rhs):
        (D,P) = self.data.shape[:2]
        if numpy.isscalar(rhs) or isinstance(rhs,numpy.ndarray):
            self.data[...] /= rhs
        else:
            retval = self.clone()
            for d in range(D):
                retval.data[d,:,...] = 1./ rhs.data[0,:,...] * ( self.data[d,:,...] - numpy.sum(retval.data[:d,:,...] * rhs.data[d:0:-1,:,...], axis=0))
            self.data[...] = retval.data[...]
        return self

    def sqrt(self):
        retval = self.clone()
        self._sqrt(self.data, out = retval.data)
        return retval
        
    def exp(self):
        retval = self.clone()
        self._exp(self.data, out = retval.data)
        return retval
        
    def log(self):
        retval = self.clone()
        self._log(self.data, out = retval.data)
        return retval

    def __abs__(self):
        """ absolute value of polynomials
        
        FIXME: theory tells us to check first coefficient if the zero'th coefficient is zero
        """
        # check if zero order coeff is smaller than 0
        tmp = self.data[0] < 0
        
        # check that taking absolute value for vectorized polynomials (i.e. P > 1) is well-defined
        D,P = self.data.shape[:2]
        for p in range(P-1):
            if (tmp[p] - tmp[p+1]).any():
                raise ValueError('vectorized version of abs works only if all directions P have the same sign!')
        
        retval = self.clone()
        retval.data *= (-1)**tmp[0]
        
        return retval



    def __neg__(self):
        return self.__class__(-self.data)
        
    @classmethod
    def add(cls, x, y , out = None):
        return x + y
        
    @classmethod
    def sub(cls, x, y , out = None):
        return x - y
        
    @classmethod
    def mul(cls, x, y , out = None):
        return x * y
        
    @classmethod
    def div(cls, x, y , out = None):
        return x / y

    @classmethod
    def multiply(cls, x, y , out = None):
        return x * y
        
    @classmethod
    def max(cls, a, axis = None, out = None):
        if out != None:
            raise NotImplementedError('should implement that')

        if axis != None:
            raise NotImplementedError('should implement that')
        
        a_shp = a.data.shape
        out_shp = a_shp[:2]
        out = cls(cls.__zeros__(out_shp, dtype = a.data.dtype))
        cls._max( a.data, axis = axis, out = out.data)
        return out

    @classmethod
    def argmax(cls, a, axis = None):
        if axis != None:
            raise NotImplementedError('should implement that')

        return cls._argmax( a.data, axis = axis)

    @classmethod
    def trace(cls, x):
        """ returns a new UTPM in standard format, i.e. the matrices are 1x1 matrices"""
        D,P = x.data.shape[:2]
        retval = numpy.zeros((D,P))
        for d in range(D):
            for p in range(P):
                retval[d,p] = numpy.trace(x.data[d,p,...])
        return UTPM(retval)
        
        
    def FtoJT(self):
        """
        Combines several directional derivatives and combines them to a transposed Jacobian JT, i.e.
        x.data.shape = (D,P,shp)
        y = x.FtoJT()
        y.data.shape = (D-1, (P,1) + shp)
        """
        D,P = self.data.shape[:2]
        shp = self.data.shape[2:]
        return UTPM(self.data[1:,...].reshape((D-1,1) + (P,) + shp))
        
    def JTtoF(self):
        """
        inverse operation of FtoJT
        x.data.shape = (D,1, P,shp)
        y = x.JTtoF()
        y.data.shape = (D+1, P, shp)
        """
        D = self.data.shape[0]
        P = self.data.shape[2]
        shp = self.data.shape[3:]
        tmp = numpy.zeros((D+1,P) + shp)
        tmp[0:D,...] = self.data.reshape((D,P) + shp)
        return UTPM(tmp)        

    def clone(self):
        """
        Returns a new UTPM instance with the same data.
        
        `clone` is opposed to `copy` or `deepcopy` by calling the __init__ function.
        
        Rationale:
            the __init__ function may have side effects that must be executed.
            Naming stems from the fact that a cloned animal is not an exact copy
            but built using the same information.
        """
        return UTPM(self.data.copy())
        
    def copy(self):
        """ this method is equivalent to `clone`.
        It's there to allow generic programming because ndarrays do not have the clone method."""
        return self.clone()

    def get_shape(self):
        return numpy.shape(self.data[0,0,...])
    shape = property(get_shape)
    
    def get_ndim(self):
        return numpy.ndim(self.data[0,0,...])
    ndim = property(get_ndim)
    
    def reshape(self, dims):
        return UTPM(self.data.reshape(self.data.shape[0:2] + dims))

    def get_transpose(self):
        return self.transpose()
    def set_transpose(self,x):
        raise NotImplementedError('???')
    T = property(get_transpose, set_transpose)

    def transpose(self, axes = None):
        return UTPM( UTPM._transpose(self.data))
        
    def get_owndata(self):
        return self.data.flags['OWNDATA']
    
    owndata = property(get_owndata)

    def set_zero(self):
        self.data[...] = 0.
        return self

    def zeros_like(self):
        return self.__class__(numpy.zeros_like(self.data))
        
    def shift(self, s, out = None):
        """
        shifting coefficients [x0,x1,x2,x3] s positions
        
        e.g. shift([x0,x1,x2,x3], -1) = [x1,x2,x3,0]
             shift([x0,x1,x2,x3], +1) = [0,x0,x1,x2]
        """
        
        if out == None:
            out = self.zeros_like()
        
        if s <= 0:
            out.data[:s,...] = self.data[-s:,...]
        
        else:
            out.data[s:,...] = self.data[:-s,...]
        
        return out
        

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return self.__str__()
        
    @classmethod
    def triu(cls, x, out = None):
        out = x.zeros_like()
        D,P = out.data.shape[:2]
        # print D,P
        for d in range(D):
            for p in range(P):
                out.data[d,p] = numpy.triu(x.data[d,p])
        
        return out
        
    
    @classmethod
    def dot(cls, x, y, out = None):
        """
        out = dot(x,y)
        
        """
        
        if isinstance(x, UTPM) and isinstance(y, UTPM):
            x_shp = x.data.shape
            y_shp = y.data.shape
            
            assert x_shp[:2] == y_shp[:2]
            
            if  len(y_shp[2:]) == 1:
                out_shp = x_shp[:-1]
                
            else:
                out_shp = x_shp[:2] + x_shp[2:-1] + y_shp[2:][:-2] + y_shp[2:][-1:]
                
            out = cls(cls.__zeros__(out_shp, dtype = x.data.dtype))
            cls._dot( x.data, y.data, out = out.data)
            
        elif isinstance(x, UTPM) and not isinstance(y, UTPM):
            x_shp = x.data.shape
            y_shp = y.shape
            
            if  len(y_shp) == 1:
                out_shp = x_shp[:-1]
                
            else:
                out_shp = x_shp[:2] + x_shp[2:-1] + y_shp[:-2] + y_shp[-1:]
                
            out = cls(cls.__zeros__(out_shp, dtype = x.data.dtype))
            cls._dot_non_UTPM_y(x.data, y, out = out.data)
            
        elif not isinstance(x, UTPM) and isinstance(y, UTPM):
            x_shp = x.shape
            y_shp = y.data.shape
            
            if  len(y_shp[2:]) == 1:
                out_shp = y_shp[:2] + x_shp[:-1]
                
            else:
                out_shp = y_shp[:2] + x_shp[:-1] + y_shp[2:][:-2] + y_shp[2:][-1:]

            out = cls(cls.__zeros__(out_shp, dtype = y.data.dtype))
            cls._dot_non_UTPM_x(x, y.data, out = out.data)
            
            
        else:
            raise NotImplementedError('should implement that')
            
        return out
    
    @classmethod
    def inv(cls, A, out = None):
        if out == None:
            out = cls(cls.__zeros__(A.data.shape, dtype = A.data.dtype))
        else:
            raise NotImplementedError('')
        
        cls._inv(A.data,(out.data,))
        return out
        # # tc[0] element
        # for p in range(P):
        #     out.data[0,p,:,:] = numpy.linalg.inv(A.data[0,p,:,:])

        # # tc[d] elements
        # for d in range(1,D):
        #     for p in range(P):
        #         for c in range(1,d+1):
        #             out.data[d,p,:,:] += numpy.dot(A.data[c,p,:,:], out.data[d-c,p,:,:],)
        #         out.data[d,p,:,:] =  numpy.dot(-out.data[0,p,:,:], out.data[d,p,:,:],)
        # return out
        
    @classmethod
    def solve(cls, A, x, out = None):
        """
        solves for y in: A y = x
        
        """
        if isinstance(A, UTPM) and isinstance(x, UTPM):
            A_shp = A.data.shape
            x_shp = x.data.shape
    
            assert A_shp[:2] == x_shp[:2]
            if A_shp[2] != x_shp[2]:
                print ValueError('A.data.shape = %s does not match x.data.shape = %s'%(str(A_shp), str(x_shp)))
    
            D, P, M = A_shp[:3]
            
            if out == None:
                out = cls(cls.__zeros__((D,P,M) + x_shp[3:], dtype = A.data.dtype))
    
            UTPM._solve(A.data, x.data, out = out.data)
        
        elif not isinstance(A, UTPM) and isinstance(x, UTPM):
            A_shp = numpy.shape(A)
            x_shp = numpy.shape(x.data)
            M = A_shp[0]
            D,P = x_shp[:2]
            out = cls(cls.__zeros__((D,P,M) + x_shp[3:], dtype = A.data.dtype))
            cls._solve_non_UTPM_A(A, x.data, out = out.data)
            
        elif isinstance(A, UTPM) and not isinstance(x, UTPM):
            A_shp = numpy.shape(A.data)
            x_shp = numpy.shape(x)
            D,P,M = A_shp[:3]
            out = cls(cls.__zeros__((D,P,M) + x_shp[1:], dtype = A.data.dtype))
            cls._solve_non_UTPM_x(A.data, x, out = out.data)
            
        else:
            raise NotImplementedError('should implement that')
            
        return out
   
    @classmethod
    def cholesky(cls, A, out = None):
        if out == None:
            out = A.zeros_like()
            
        cls._cholesky(A.data, out.data)
        return out


    @classmethod
    def pb_cholesky(cls, Lbar, A, L, out = None):
        if out == None:
            D,P = A.data.shape[:2]
            Abar = A.zeros_like()
        
        else:
            Abar = out
            
        cls._pb_cholesky(Lbar.data, A.data, L.data, out = Abar.data)
        return Abar
        

    @classmethod
    def pb_Id(cls, ybar, x, y, out = None):
        return out
        
        
    @classmethod
    def pb___add__(cls, zbar, x, y , z, out = None):
        return cls.pb_add(zbar, x, y , z, out = out)
        
    @classmethod
    def pb___sub__(cls, zbar, x, y , z, out = None):
        return cls.pb_sub(zbar, x, y , z, out = out)
        
    @classmethod
    def pb___mul__(cls, zbar, x, y , z, out = None):
        return cls.pb_mul(zbar, x, y , z, out = out)
        
    @classmethod
    def pb___div__(cls, zbar, x, y , z, out = None):
        return cls.pb_div(zbar, x, y , z, out = out)
        
    @classmethod
    def pb_add(cls, zbar, x, y , z, out = None):
        if out == None:
            D,P = y.data.shape[:2]
            xbar = x.zeros_like()
            ybar = y.zeros_like()
        
        else:
            xbar, ybar = out
        
        ybar += zbar
        xbar += zbar

        return (xbar,ybar)
        
        
    @classmethod
    def pb___iadd__(cls, zbar, x, y, z, out = None):
        # FIXME: this is a workaround/hack, review this
        x = x.copy()
        return cls.pb___add__(zbar, x, y, z, out = out)
        # if out == None:
            # D,P = y.data.shape[:2]
            # xbar = cls(cls.__zeros__(x.data.shape))
            # ybar = cls(cls.__zeros__(y.data.shape))
        
        # else:
            # xbar, ybar = out
        
        # xbar = zbar
        # ybar += zbar
        
    
        # return xbar, ybar
        
    @classmethod
    def pb_sub(cls, zbar, x, y , z, out = None):
        if out == None:
            D,P = y.data.shape[:2]
            xbar = x.zeros_like()
            ybar = y.zeros_like()
        
        else:
            xbar, ybar = out
        
        xbar += zbar
        ybar -= zbar

        return (xbar,ybar)


    @classmethod
    def pb_mul(cls, zbar, x, y , z, out = None):
        if out == None:
            D,P = y.data.shape[:2]
            xbar = x.zeros_like()
            ybar = y.zeros_like()
        
        else:
            xbar, ybar = out
        
        xbar += zbar * y
        ybar += zbar * x

        return (xbar,ybar)
        
    @classmethod
    def pb_div(cls, zbar, x, y , z, out = None):
        if out == None:
            D,P = y.data.shape[:2]
            xbar = x.zeros_like()
            ybar = y.zeros_like()
        
        else:
            xbar, ybar = out
        
        tmp  = zbar.clone()
        tmp /= y
        xbar += tmp
        tmp *= z
        ybar -= tmp

        return (xbar,ybar)
        
    @classmethod
    def pb_dot(cls, zbar, x, y, z, out = None):
        if out == None:
            D,P = y.data.shape[:2]
            xbar = x.zeros_like()
            ybar = y.zeros_like()
        
        else:
            xbar, ybar = out
        
        cls._dot_pullback(zbar.data, x.data, y.data, z.data, out = (xbar.data, ybar.data))
        return (xbar,ybar)
        
    @classmethod
    def pb_inv(cls, ybar, x, y, out = None):
        if out == None:
            D,P = y.data.shape[:2]
            xbar = x.zeros_like()
        
        else:
            xbar, = out
            
        cls._inv_pullback(ybar.data, x.data, y.data, out = xbar.data)
        return xbar

        
    @classmethod
    def pb_solve(cls, ybar, A, x, y, out = None):
        D,P = y.data.shape[:2]
        
        
        if not isinstance(A, UTPM):
            raise NotImplementedError('should implement that')
        
        if not isinstance(x, UTPM):
            
            tmp = x
            x = UTPM(numpy.zeros( (D,P) + x.shape))
            for p in range(P):
                x.data[0,p] = tmp[...]
            
        if out == None:
            xbar = x.zeros_like()
            Abar = A.zeros_like()
        
        else:
            Abar, xbar = out

        cls._solve_pullback(ybar.data, A.data, x.data, y.data, out = (Abar.data, xbar.data))

        return Abar, xbar

    @classmethod
    def pb_trace(cls, ybar, x, y, out = None):
        if out == None:
            out = (x.zeros_like(),)
        
        xbar, = out
        Nx = xbar.shape[0]
        for nx in range(Nx):
            xbar[nx,nx] += ybar
        
        return xbar

    @classmethod
    def pb_transpose(cls, ybar, x, y, out = None):
        if out == None:
            raise NotImplementedError('should implement that')
        
        xbar, = out
        xbar = cls.transpose(ybar)
        return xbar

    @classmethod
    def qr(cls, A, out = None, work = None):
        D,P,M,N = numpy.shape(A.data)
        K = min(M,N)
        
        if out == None:
            Q = cls(cls.__zeros__((D,P,M,K), dtype=A.data.dtype))
            R = cls(cls.__zeros__((D,P,K,N), dtype=A.data.dtype))
            
        else:
            Q,R = out
        
        UTPM._qr(A.data, out = (Q.data, R.data))
        
        return Q,R
        
    @classmethod
    def pb_qr(cls, Qbar, Rbar, A, Q, R, out = None):
        D,P,M,N = numpy.shape(A.data)
        
        if out == None:
            Abar = A.zeros_like()
        
        else:
            Abar, = out
        
        UTPM._qr_pullback( Qbar.data, Rbar.data, A.data, Q.data, R.data, out = Abar.data)
        return Abar

    
    @classmethod
    def eigh(cls, A, out = None, epsilon = 10**-8):
        """
        computes the eigenvalue decomposition A = Q^T L Q
        of a symmetrical matrix A with distinct eigenvalues
        
        (l,Q) = UTPM.eig(A, out=None)
        
        """
        
        D,P,M,N = numpy.shape(A.data)
        
        if out == None:
            l = cls(cls.__zeros__((D,P,N), dtype=A.data.dtype))
            Q = cls(cls.__zeros__((D,P,N,N), dtype=A.data.dtype))
            
        else:
            l,Q = out
        
        UTPM._eigh( l.data, Q.data, A.data, epsilon = epsilon)
      
        return l,Q

    @classmethod
    def pb_eigh(cls, lbar, Qbar,  A, l, Q,  out = None):
        D,P,M,N = numpy.shape(A.data)
        
        if out == None:
            Abar = A.zeros_like()
        
        else:
            Abar, = out
        
        UTPM._eigh_pullback( lbar.data,  Qbar.data, A.data,  l.data, Q.data, out = Abar.data)
        return Abar

    @classmethod
    def diag(cls, v, k = 0, out = None):
        """Extract a diagonal or construct  diagonal UTPM instance"""
        return cls(cls._diag(v.data))
    
    @classmethod
    def iouter(cls, x, y, out):
        cls._iouter(x.data, y.data, out.data)
        return out

    @classmethod
    def reshape(cls, a, newshape, order = 'C'):

        if order != 'C':
            raise NotImplementedError('should implement that')
        
        return cls(cls._reshape(a.data, newshape, order = order))
    
    @classmethod
    def combine_blocks(cls, in_X):
        """
        expects an array or list consisting of entries of type UTPM, e.g.
        in_X = [[UTPM1,UTPM2],[UTPM3,UTPM4]]
        and returns
        UTPM([[UTPM1.data,UTPM2.data],[UTPM3.data,UTPM4.data]])
    
        """
    
        in_X = numpy.array(in_X)
        Rb,Cb = numpy.shape(in_X)
    
        # find the degree D and number of directions P
        D = 0; 	P = 0;
    
        for r in range(Rb):
            for c in range(Cb):
                D = max(D, in_X[r,c].data.shape[0])
                P = max(P, in_X[r,c].data.shape[1])
    
        # find the sizes of the blocks
        rows = []
        cols = []
        for r in range(Rb):
            rows.append(in_X[r,0].shape[0])
        for c in range(Cb):
            cols.append(in_X[0,c].shape[1])
        rowsums = numpy.array([ numpy.sum(rows[:r]) for r in range(0,Rb+1)],dtype=int)
        colsums = numpy.array([ numpy.sum(cols[:c]) for c in range(0,Cb+1)],dtype=int)
    
        # create new matrix where the blocks will be copied into
        tc = numpy.zeros((D, P, rowsums[-1],colsums[-1]))
        for r in range(Rb):
            for c in range(Cb):
                tc[:,:,rowsums[r]:rowsums[r+1], colsums[c]:colsums[c+1]] = in_X[r,c].data[:,:,:,:]
    
        return UTPM(tc) 
