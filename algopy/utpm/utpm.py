"""
Implementation of the univariate matrix polynomial.
The algebraic class is

M[t]/<t^D>

where M is the ring of matrices and t an external parameter

"""

import numpy.linalg
import numpy

from ..base_type import Ring
from algorithms import RawAlgorithmsMixIn, broadcast_arrays_shape

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
    def as_utpm(cls, x):
        """ tries to convert a container (e.g. list or numpy.array) with UTPM elements as instances to a UTPM instance"""
        
        x_shp = numpy.shape(x)
        xr = numpy.ravel(x)
        D,P = xr[0].data.shape[:2]
        shp = xr[0].data.shape[2:]
        
        if not isinstance(shp, tuple): shp = (shp,)
        if not isinstance(x_shp, tuple): x_shp = (x_shp,)
        
        y = UTPM(numpy.zeros((D,P) + x_shp + shp))
        
        yr = UTPM( y.data.reshape((D,P) + (numpy.prod(x_shp),) + shp))
        
        # print yr.shape
        # print yr.data.shape
        
        for n in range(len(xr)):
            # print yr[n].shape
            # print xr[n].shape
            yr[n] = xr[n]
            
        return y
            
        
    def get_flat(self):
        return UTPM(self.data.reshape(self.data.shape[:2] + (numpy.prod(self.data.shape[2:]),) ))
        
    flat = property(get_flat)
    
    
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
    
    def __add__(self,rhs):
        if numpy.isscalar(rhs):
            retval = UTPM(numpy.copy(self.data))
            retval.data[0,:] += rhs
            return retval
        elif isinstance(rhs, numpy.ndarray):
            rhs_shape = rhs.shape
            if numpy.isscalar(rhs_shape):
                rhs_shape = (rhs_shape,)
            x_data, y_data = UTPM._broadcast_arrays(self.data, rhs.reshape((1,1)+rhs_shape))
            z_data = x_data.copy()
            z_data[0] += y_data[0]
            return UTPM(z_data)

        else:
            return UTPM(self.data + rhs.data)

            
    def __sub__(self,rhs):
        if numpy.isscalar(rhs):
            retval = UTPM(numpy.copy(self.data))
            retval.data[0,:] -= rhs
            return retval
            
        elif isinstance(rhs, numpy.ndarray):
            rhs_shape = rhs.shape
            if numpy.isscalar(rhs_shape):
                rhs_shape = (rhs_shape,)
            x_data, y_data = UTPM._broadcast_arrays(self.data, rhs.reshape((1,1)+rhs_shape))
            z_data = x_data.copy()
            z_data[0] -= y_data[0]
            return UTPM(z_data)         
            
        else:
            return UTPM(self.data - rhs.data)

    def __mul__(self,rhs):
        if numpy.isscalar(rhs):
            return UTPM( self.data * rhs)

        elif isinstance(rhs,numpy.ndarray):
            rhs_shape = rhs.shape
            if numpy.isscalar(rhs_shape):
                rhs_shape = (rhs_shape,)
            return UTPM( (self.data.T * rhs.reshape((1,1) + rhs_shape).T).T)

        
        x_data, y_data = UTPM._broadcast_arrays(self.data, rhs.data)
        z_data = numpy.zeros_like(x_data)
        self._mul(x_data, y_data, z_data)
        return self.__class__(z_data)

    def __div__(self,rhs):
        if numpy.isscalar(rhs):
            return UTPM( self.data/rhs)
            
        elif isinstance(rhs, numpy.ndarray):
            rhs_shape = rhs.shape
            if numpy.isscalar(rhs_shape):
                rhs_shape = (rhs_shape,)
            return UTPM((self.data.T / rhs.reshape((1,1) + rhs_shape ).T).T )            
        
        x_data, y_data = UTPM._broadcast_arrays(self.data, rhs.data)
        z_data = numpy.zeros_like(x_data)
        self._div(x_data, y_data, z_data)
        return self.__class__(z_data)
        
    def __pow__(self,r):
        if isinstance(r, UTPM):
            return numpy.exp(numpy.log(self)*r)
        else:
            x_data = self.data
            y_data = numpy.zeros_like(x_data)
            self._pow_real(x_data, r, y_data)
            return self.__class__(y_data)
        

    def __radd__(self,rhs):
        return self + rhs

    def __rsub__(self, other):
        return -self + other

    def __rmul__(self,rhs):
        return self * rhs

    def __rdiv__(self, rhs):
        tmp = self.zeros_like()
        tmp.data[0,...] = rhs
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
        
    def sincos(self):
        retsin = self.clone()
        retcos = self.clone()
        self._sincos(self.data, out = (retsin.data, retcos.data))
        return retsin, retcos        

    def sum(self, axis=None, dtype=None, out=None):
        if dtype != None or out != None:
            raise NotImplementedError('not implemented yet')
        
        if axis == None:
            tmp = numpy.prod(self.data.shape[2:])
            return UTPM(numpy.sum(self.data.reshape(self.data.shape[:2] + (tmp,)), axis = 2))
        else:
            return UTPM(numpy.sum(self.data, axis = axis + 2))


    @classmethod
    def pb_sincos(cls, sbar, cbar, x, s, c, out = None):
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()
        
        else:
            xbar, = out
            
        cls._pb_sincos(sbar.data, cbar.data, x.data, s.data, c.data, out = xbar.data)
        
        return out        
        
    def sin(self):
        retval = self.clone()
        tmp = self.clone()
        self._sincos(self.data, out = (retval.data, tmp.data))
        return retval
        
    @classmethod
    def pb_sin(cls, sbar, x, s,  out = None):
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()
        
        else:
            xbar, = out
            
        c = x.cos()
        cbar = x.zeros_like()
        cls._pb_sincos(sbar.data, cbar.data, x.data, s.data, c.data, out = xbar.data)
        return out
        
    def cos(self):
        retval = self.clone()
        tmp = self.clone()
        self._sincos(self.data, out = (tmp.data, retval.data))
        return retval
        
    @classmethod
    def pb_cos(cls, cbar, x, c,  out = None):
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()
        
        else:
            xbar, = out
            
        s = x.sin()
        sbar = x.zeros_like()
        cls._pb_sincos(sbar.data, cbar.data, x.data, s.data, c.data, out = xbar.data)
        return out        
        

            

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
        return self.__class__.neg(self)
        
    @classmethod
    def neg(cls, x, out = None):
        return -1*x
        
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
    
    
    def get_size(self):
        return numpy.size(self.data[0,0,...])
    size = property(get_size)
    
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
    def pb_neg(cls, ybar, x, y, out = None):
        if out == None:
            xbar = x.zeros_like()
        
        else:
            xbar, = out
            
        xbar -= ybar
        return xbar
        
    @classmethod
    def pb___neg__(cls, ybar, x, y, out = None):
        return cls.pb_neg(ybar, x, y, out = out)

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
    def broadcast(cls, x,y):
        if numpy.isscalar(x) or isinstance(x,numpy.ndarray):
            return x,y
        
        if numpy.isscalar(y) or isinstance(y,numpy.ndarray):
            return x,y
            
        # broadcast xbar and ybar
        x2_data, y2_data = cls._broadcast_arrays(x.data,y.data)
        
        x2 = UTPM(x2_data)
        y2 = UTPM(y2_data)
        return x2, y2

    @classmethod
    def pb_mul(cls, zbar, x, y , z, out = None):
        if out == None:
            D,P = y.data.shape[:2]
            xbar = x.zeros_like()
            ybar = y.zeros_like()
        
        else:
            xbar, ybar = out
            
        xbar2, tmp = cls.broadcast(xbar, zbar)
        ybar2, tmp = cls.broadcast(ybar, zbar)
        
        xbar2 += zbar * y
        ybar2 += zbar * x

        return (xbar,ybar)
        
    @classmethod
    def pb_div(cls, zbar, x, y , z, out = None):
        if out == None:
            D,P = y.data.shape[:2]
            xbar = x.zeros_like()
            ybar = y.zeros_like()
        
        else:
            xbar, ybar = out
            
        xbar2, tmp = cls.broadcast(xbar, zbar)
        ybar2, tmp = cls.broadcast(ybar, zbar)            
        
        tmp  = zbar.clone()
        tmp /= y
        xbar2 += tmp
        tmp *= z
        ybar2 -= tmp

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
            if out[1] == None:
                Abar = out[0]
                xbar = x.zeros_like()
            
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
    def qr(cls, A, out = None, work = None, epsilon = 10**-14):
        D,P,M,N = numpy.shape(A.data)
        K = min(M,N)
        
        if out == None:
            Q = cls(cls.__zeros__((D,P,M,K), dtype=A.data.dtype))
            R = cls(cls.__zeros__((D,P,K,N), dtype=A.data.dtype))
            
        else:
            Q,R = out
        
        UTPM._qr(A.data, out = (Q.data, R.data), epsilon = epsilon)
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
    def qr_full(cls, A, out = None, work = None):
        D,P,M,N = numpy.shape(A.data)
        
        if out == None:
            Q = cls(cls.__zeros__((D,P,M,M), dtype=A.data.dtype))
            R = cls(cls.__zeros__((D,P,M,N), dtype=A.data.dtype))
            
        else:
            Q,R = out
        
        UTPM._qr_full(A.data, out = (Q.data, R.data))
        
        return Q,R
        
    @classmethod
    def pb_qr_full(cls, Qbar, Rbar, A, Q, R, out = None):
        D,P,M,N = numpy.shape(A.data)
        
        if out == None:
            Abar = A.zeros_like()
        
        else:
            Abar, = out
        
        UTPM._qr_full_pullback( Qbar.data, Rbar.data, A.data, Q.data, R.data, out = Abar.data)
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
    def eigh1(cls, A, out = None, epsilon = 10**-8):
        """
        computes the relaxed eigenvalue decompositin of level 1
        of a symmetrical matrix A with distinct eigenvalues
        
        (L,Q,b) = UTPM.eig1(A)
        
        """
        
        D,P,M,N = numpy.shape(A.data)
        
        if out == None:
            L = cls(cls.__zeros__((D,P,N,N), dtype=A.data.dtype))
            Q = cls(cls.__zeros__((D,P,N,N), dtype=A.data.dtype))
            
        else:
            L,Q = out
        
        b_list = []
        for p in range(P):
            b = UTPM._eigh1( L.data[:,p], Q.data[:,p], A.data[:,p], epsilon = epsilon)
            b_list.append(b)
      
        return L,Q,b_list
        
        
        

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
    def pb_eigh1(cls, Lbar, Qbar, bbar_list, A, L, Q, b_list,  out = None):
        D,P,M,N = numpy.shape(A.data)
        
        if out == None:
            Abar = A.zeros_like()
        
        else:
            Abar, = out
        
        UTPM._eigh1_pullback( Lbar.data,  Qbar.data, A.data,  L.data, Q.data, b_list, out = Abar.data)
        return Abar

    @classmethod
    def diag(cls, v, k = 0, out = None):
        """Extract a diagonal or construct  diagonal UTPM instance"""
        return cls(cls._diag(v.data))
        
    @classmethod
    def pb_diag(cls, ybar, x, y, k = 0, out = None):
        """Extract a diagonal or construct  diagonal UTPM instance"""
        
        if out == None:
            xbar = x.zeros_like()
        
        else:
            xbar, = out

        return cls(cls._diag_pullback(ybar.data, x.data, y.data, k = k, out = xbar.data))        


    @classmethod
    def symvec(cls, A):
        """
        maps a symmetric matrix to a vector containing the distinct elements
        """
        D,P,N,M = A.data.shape
        assert N == M
        
        v = cls(numpy.zeros( (D,P,((N+1)*N)//2)))
        
        count = 0
        for row in range(N):
            for col in range(row,N):
                v[count] = 0.5* (A[row,col] + A[col,row])
                count +=1
        return v
            
    @classmethod
    def pb_symvec(cls, vbar, A, v, out = None):
        
        if out == None:
            Abar = A.zeros_like()
        
        else:
            Abar ,= out
        
        Abar += cls.vecsym(vbar)
        return Abar
            
    @classmethod
    def vecsym(cls, v):
        """
        returns a full symmetric matrix filled
        the distinct elements of v, filled row-wise
        """
        D,P = v.data.shape[:2]
        Nv = v.data[0,0].size
        
        tmp = numpy.sqrt(1 + 8*Nv)
        if abs(int(tmp) - tmp) > 10**-16:
            # hackish way to check that the input length of v makes sense
            raise ValueError('size of v does not match any possible symmetric matrix')
        N = (int(tmp) - 1)//2
        A = cls(numpy.zeros((D,P,N,N)))
        
        count = 0
        for row in range(N):
            for col in range(row,N):
                A[row,col] = A[col,row] = v[count]
                count +=1
        
        return A
            
    @classmethod
    def pb_vecsym(cls, Abar, v, A, out = None):
        
        if out == None:
            vbar = v.zeros_like()
        
        else:
            vbar ,= out
        
        vbar += cls.symvec(Abar)
        return vbar
                

        
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
