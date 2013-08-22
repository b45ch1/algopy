"""
Implementation of the univariate matrix polynomial.
The algebraic class is

M[t]/<t^D>

where M is the ring of matrices and t an external parameter

"""

import math

import numpy.linalg
import numpy

from ..base_type import Ring

from .algorithms import RawAlgorithmsMixIn, broadcast_arrays_shape

import operator

from algopy import nthderiv


if float(numpy.__version__[:3]) > 1.5:

    def workaround_strides_function(x, y, fun):
        """

        peform the operation fun(x,y)

        where fun = operator.iadd, operator.imul, operator.setitem, etc.

        workaround for the bug
        https://github.com/numpy/numpy/issues/2705

        Replace this function once the bug has been fixed.

        This function assumes that x and y have the same shape.


        Parameters
        ------------

        x:      UTPM instance

        y:      UTPM instance

        fun:    function from the module operator

        """

        if x.shape != y.shape:
            raise ValueError('x.shape != y.shape')

        if x.ndim == 0:
            fun(x, y)
        else:
            for i in range(x.shape[0]):
                workaround_strides_function(x[i, ...], y[i, ...], fun)

else:

    def workaround_strides_function(x, y, fun):
        fun(x, y)


class UTPM(Ring, RawAlgorithmsMixIn):
    """

    UTPM == Univariate Taylor Polynomial of Matrices
    This class implements univariate Taylor arithmetic on matrices, i.e.
    [A]_D = \sum_{d=0}^{D-1} A_d T^d

    Input:
    in the most general form, the input is a 4-tensor.
    We use the notation:
    D: degree of the Taylor series
    P: number of directions
    N: number of rows of A_0
    M: number of cols of A_0

    shape([A]) = (D,P,N,M)
    The reason for this choice is that the (N,M) matrix is the elementary type,
    so that memory should be contiguous.
    Then, at each operation, the code performed to compute
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

    For ufuncs this arrangement is advantageous, because in this order,
    memory chunks of size Ndir are used and the operation on each element is the
    same. This is desireable to avoid cache misses.
    See for example __mul__: there, operations of

    self.data[:d+1,:,:,:]* rhs.data[d::-1,:,:,:]

    has to be performed.
    One can see, that contiguous memory blocks are used for such operations.

    A disadvantage of this arrangement is: it seems unnatural.
    It is easier to regard each direction separately.
    """

    __array_priority__ = 2

    def __init__(self, X):
        """

        INPUT:
        shape([X]) = (D,P,N,M)
        """
        Ndim = numpy.ndim(X)
        if Ndim >= 2:
            self.data = numpy.asarray(X)
            self.data = self.data
        else:
            raise NotImplementedError

    def __getitem__(self, sl):
        if isinstance(sl, int) or sl == Ellipsis or isinstance(sl, slice):
            sl = (sl,)

        tmp = self.data.__getitem__((slice(None),slice(None)) + tuple(sl))
        return self.__class__(tmp)

    def __setitem__(self, sl, rhs):
        if isinstance(rhs, UTPM):
            if type(sl) == int or sl == Ellipsis or isinstance(sl, slice):
                sl = (sl,)
            x_data, y_data = UTPM._broadcast_arrays(self.data.__getitem__((slice(None),slice(None)) + sl), rhs.data)
            return x_data.__setitem__(Ellipsis, y_data)
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
            raise NotImplementedError('I\'m not sure that this makes sense')

        # workaround for qr and eigh
        if isinstance( out[0], tuple):
            tmp = list(out[0])
            tmp[sl] += ybar

        # usual workflow
        else:
            # print 'out=\n', out[0][sl]
            # print 'ybar=\n',ybar
            out[0][sl] = ybar

        return out

    @classmethod
    def pb_getitem(cls, ybar, x, sl, y, out = None):
        # print 'ybar=\n',ybar
        retval = cls.pb___getitem__(ybar, x, sl, y, out = out)
        # print 'retval=\n',retval[0]
        return retval


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

    def coeff_op(self, sl, shp):
        """
        operation to extract UTP coefficients of x
        defined by the slice sl creates a new
        UTPM instance where the coefficients have the shape as defined
        by shp

        Parameters
        ----------
        x: UTPM instance
        sl: tuple of slice instance
        shp: tuple

        Returns
        -------
        UTPM instance

        """

        tmp = self.data.__getitem__(sl)
        tmp = tmp.reshape(shp)
        return self.__class__(tmp)

    @classmethod
    def pb_coeff_op(cls, ybar, x, sl, shp, out = None):

        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar = out[0]

        # step 1: revert reshape
        old_shp = x.data.__getitem__(sl).shape
        tmp_data = ybar.data.reshape(old_shp)

        print('tmp_data.shape=',tmp_data.shape)

        # step 2: revert getitem
        tmp2 = xbar.data[::-1].__getitem__(sl)
        tmp2 += tmp_data[::-1,...]

        return xbar



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
    def pb_setitem(cls, y, sl, x, out = None):
        return cls.pb___setitem__(y, sl, x, out = out)

    def __add__(self,rhs):
        if numpy.isscalar(rhs):
            retval = UTPM(numpy.copy(self.data))
            retval.data[0,:] += rhs
            return retval

        elif isinstance(rhs,numpy.ndarray) and rhs.dtype == object:

            if not isinstance(rhs.flatten()[0], UTPM):
                err_str = 'you are trying to add a UTPM instance with a numpy.ndarray with elements of type %s\n'%type(rhs.flatten()[0])
                err_str+= 'this operation is not supported!\n'
                raise NotImplementedError(err_str)
            else:
                err_str = 'should implement that!!!'
                raise NotImplementedError(err_str)

        elif isinstance(rhs, numpy.ndarray):
            rhs_shape = rhs.shape
            if numpy.isscalar(rhs_shape):
                rhs_shape = (rhs_shape,)
            x_data, y_data = UTPM._broadcast_arrays(self.data, rhs.reshape((1,1)+rhs_shape))
            z_data = x_data.copy()
            z_data[0] += y_data[0]
            return UTPM(z_data)

        else:
            x_data, y_data = UTPM._broadcast_arrays(self.data, rhs.data)
            z_data = x_data.copy()
            z_data += y_data
            return UTPM(z_data)

    def __sub__(self,rhs):
        if numpy.isscalar(rhs):
            retval = UTPM(numpy.copy(self.data))
            retval.data[0,:] -= rhs
            return retval

        elif isinstance(rhs,numpy.ndarray) and rhs.dtype == object:
            raise NotImplementedError('should implement that')

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

        elif isinstance(rhs,numpy.ndarray) and rhs.dtype == object:
            raise NotImplementedError('should implement that')

        elif isinstance(rhs,numpy.ndarray):
            rhs_shape = rhs.shape
            if numpy.isscalar(rhs_shape):
                rhs_shape = (rhs_shape,)
            x_data, y_data = UTPM._broadcast_arrays(self.data, rhs.reshape((1,1)+rhs_shape))
            return UTPM(x_data * y_data)

        x_data, y_data = UTPM._broadcast_arrays(self.data, rhs.data)
        z_data = numpy.zeros_like(x_data)
        self._mul(x_data, y_data, z_data)
        return self.__class__(z_data)

    def __truediv__(self,rhs):
        if numpy.isscalar(rhs):
            return UTPM( self.data/rhs)

        elif isinstance(rhs,numpy.ndarray) and rhs.dtype == object:
            raise NotImplementedError('should implement that')

        elif isinstance(rhs,numpy.ndarray):
            rhs_shape = rhs.shape
            if numpy.isscalar(rhs_shape):
                rhs_shape = (rhs_shape,)
            x_data, y_data = UTPM._broadcast_arrays(self.data, rhs.reshape((1,1)+rhs_shape))
            return UTPM(x_data / y_data)

        x_data, y_data = UTPM._broadcast_arrays(self.data, rhs.data)
        z_data = numpy.zeros_like(x_data)
        self._truediv(x_data, y_data, z_data)
        return self.__class__(z_data)

    def __floordiv__(self, rhs):
        """
        self // rhs

        use L'Hopital's rule
        """

        x_data, y_data = UTPM._broadcast_arrays(self.data, rhs.data)
        z_data = numpy.zeros_like(x_data)
        self._floordiv(x_data, y_data, z_data)
        return self.__class__(z_data)


    def __pow__(self,r):
        if isinstance(r, UTPM):
            return numpy.exp(numpy.log(self)*r)
        else:
            x_data = self.data
            y_data = numpy.zeros_like(x_data)
            self._pow_real(x_data, r, y_data)
            return self.__class__(y_data)

    @classmethod
    def pb___pow__(cls, ybar, x, r, y, out = None):
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar = out[0]

        cls._pb_pow_real(ybar.data, x.data, r, y.data, out = xbar.data)
        return xbar

    @classmethod
    def pb_pow(cls, ybar, x, r, y, out = None):
        retval = cls.pb___pow__(ybar, x, r, y, out = out)
        return retval


    def __radd__(self,rhs):
        return self + rhs

    def __rsub__(self, other):
        return -self + other

    def __rmul__(self,rhs):
        return self * rhs

    def __rtruediv__(self, rhs):
        tmp = self.zeros_like()
        tmp.data[0,...] = rhs
        return tmp/self

    def __iadd__(self,rhs):
        if isinstance(rhs,numpy.ndarray) and rhs.dtype == object:
            raise NotImplementedError('should implement that')

        elif numpy.isscalar(rhs) or isinstance(rhs,numpy.ndarray):
            self.data[0,...] += rhs
        else:
            self_data, rhs_data = UTPM._broadcast_arrays(self.data, rhs.data)
            self_data[...] += rhs_data[...]
        return self

    def __isub__(self,rhs):
        if isinstance(rhs,numpy.ndarray) and rhs.dtype == object:
            raise NotImplementedError('should implement that')

        elif numpy.isscalar(rhs) or isinstance(rhs,numpy.ndarray):
            self.data[0,...] -= rhs
        else:
            self_data, rhs_data = UTPM._broadcast_arrays(self.data, rhs.data)
            self_data[...] -= rhs_data[...]
        return self

    def __imul__(self,rhs):
        (D,P) = self.data.shape[:2]

        if isinstance(rhs,numpy.ndarray) and rhs.dtype == object:
            raise NotImplementedError('should implement that')

        elif numpy.isscalar(rhs) or isinstance(rhs,numpy.ndarray):
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

    def __itruediv__(self,rhs):
        (D,P) = self.data.shape[:2]
        if isinstance(rhs,numpy.ndarray) and rhs.dtype == object:
            raise NotImplementedError('should implement that')

        elif numpy.isscalar(rhs) or isinstance(rhs,numpy.ndarray):
            self.data[...] /= rhs
        else:
            retval = self.clone()
            for d in range(D):
                retval.data[d,:,...] = 1./ rhs.data[0,:,...] * ( self.data[d,:,...] - numpy.sum(retval.data[:d,:,...] * rhs.data[d:0:-1,:,...], axis=0))
            self.data[...] = retval.data[...]
        return self

    __div__ = __truediv__
    __idiv__ = __itruediv__
    __rdiv__ = __rtruediv__

    def sqrt(self):
        retval = self.clone()
        self._sqrt(self.data, out = retval.data)
        return retval

    @classmethod
    def pb_sqrt(cls, ybar, x, y, out=None):
        """ computes bar y dy = bar x dx in UTP arithmetic"""
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar, = out

        cls._pb_sqrt(ybar.data, x.data, y.data, out = xbar.data)
        return out



    def exp(self):
        """ computes y = exp(x) in UTP arithmetic"""

        retval = self.clone()
        self._exp(self.data, out = retval.data)
        return retval

    @classmethod
    def pb_exp(cls, ybar, x, y, out=None):
        """ computes bar y dy = bar x dx in UTP arithmetic"""
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar, = out

        cls._pb_exp(ybar.data, x.data, y.data, out = xbar.data)
        return out

    def expm1(self):
        """ computes y = expm1(x) in UTP arithmetic"""

        retval = self.clone()
        self._expm1(self.data, out = retval.data)
        return retval

    @classmethod
    def pb_expm1(cls, ybar, x, y, out=None):
        """ computes bar y dy = bar x dx in UTP arithmetic"""
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar, = out

        cls._pb_expm1(ybar.data, x.data, y.data, out = xbar.data)
        return out

    def log(self):
        """ computes y = log(x) in UTP arithmetic"""
        retval = self.clone()
        self._log(self.data, out = retval.data)
        return retval

    @classmethod
    def pb_log(cls, ybar, x, y, out=None):
        """ computes bar y dy = bar x dx in UTP arithmetic"""
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar, = out

        cls._pb_log(ybar.data, x.data, y.data, out = xbar.data)
        return out

    def log1p(self):
        """ computes y = log1p(x) in UTP arithmetic"""
        retval = self.clone()
        self._log1p(self.data, out = retval.data)
        return retval

    @classmethod
    def pb_log1p(cls, ybar, x, y, out=None):
        """ computes bar y dy = bar x dx in UTP arithmetic"""
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar, = out

        cls._pb_log1p(ybar.data, x.data, y.data, out = xbar.data)
        return out

    def sincos(self):
        """ simultanteously computes s = sin(x) and c = cos(x) in UTP arithmetic"""
        retsin = self.clone()
        retcos = self.clone()
        self._sincos(self.data, out = (retsin.data, retcos.data))
        return retsin, retcos

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


    def tansec2(self):
        """ computes simultaneously y = tan(x) and z = sec^2(x)  in UTP arithmetic"""
        rettan = self.clone()
        retsec = self.clone()
        self._tansec2(self.data, out = (rettan.data, retsec.data))
        return rettan, retset

    def tan(self):
        retval = self.zeros_like()
        tmp = self.zeros_like()
        self._tansec2(self.data, out = (retval.data, tmp.data))
        return retval

    @classmethod
    def pb_tan(cls, ybar, x, y,  out = None):
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar, = out

        z = 1./x.cos(); z = z * z
        zbar = x.zeros_like()
        cls._pb_tansec(ybar.data, zbar.data, x.data, y.data, z.data, out = xbar.data)
        return out

    @classmethod
    def dpm_hyp1f1(cls, a, b, x):
        """ computes y = hyp1f1(a, b, x) in UTP arithmetic"""

        retval = x.clone()
        cls._dpm_hyp1f1(a, b, x.data, out = retval.data)
        return retval

    @classmethod
    def pb_dpm_hyp1f1(cls, ybar, a, b, x, y, out=None):
        """ computes bar y dy = bar x dx in UTP arithmetic"""
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            # out = (abar, bbar, xbar)
            xbar = out[2]

        cls._pb_dpm_hyp1f1(ybar.data, a, b, x.data, y.data, out = xbar.data)

        return xbar

    @classmethod
    def hyp1f1(cls, a, b, x):
        """ computes y = hyp1f1(a, b, x) in UTP arithmetic"""

        retval = x.clone()
        cls._hyp1f1(a, b, x.data, out = retval.data)
        return retval

    @classmethod
    def pb_hyp1f1(cls, ybar, a, b, x, y, out=None):
        """ computes bar y dy = bar x dx in UTP arithmetic"""
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            # out = (abar, bbar, xbar)
            xbar = out[2]

        cls._pb_hyp1f1(ybar.data, a, b, x.data, y.data, out = xbar.data)

        return xbar

    @classmethod
    def hyperu(cls, a, b, x):
        """ computes y = hyperu(a, b, x) in UTP arithmetic"""
        retval = x.clone()
        cls._hyperu(a, b, x.data, out = retval.data)
        return retval

    @classmethod
    def pb_hyperu(cls, ybar, a, b, x, y, out=None):
        """ computes bar y dy = bar x dx in UTP arithmetic"""
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()
        else:
            # out = (abar, bbar, xbar)
            xbar = out[2]
        cls._pb_hyperu(ybar.data, a, b, x.data, y.data, out = xbar.data)
        return xbar

    @classmethod
    def botched_clip(cls, a_min, a_max, x):
        """ computes y = botched_clip(a_min, a_max, x) in UTP arithmetic"""
        retval = x.clone()
        cls._botched_clip(a_min, a_max, x.data, out = retval.data)
        return retval

    @classmethod
    def pb_botched_clip(cls, ybar, a_min, a_max, x, y, out=None):
        """ computes bar y dy = bar x dx in UTP arithmetic"""
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()
        else:
            # out = (aminbar, amaxbar, xbar)
            xbar = out[2]
        cls._pb_botched_clip(
                ybar.data, a_min, a_max, x.data, y.data, out = xbar.data)
        return xbar

    @classmethod
    def dpm_hyp2f0(cls, a1, a2, x):
        """ computes y = hyp2f0(a1, a2, x) in UTP arithmetic"""

        retval = x.clone()
        cls._dpm_hyp2f0(a1, a2, x.data, out = retval.data)
        return retval

    @classmethod
    def pb_dpm_hyp2f0(cls, ybar, a1, a2, x, y, out=None):
        """ computes bar y dy = bar x dx in UTP arithmetic"""
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            # out = (a1bar, a2bar, xbar)
            xbar = out[2]

        cls._pb_dpm_hyp2f0(ybar.data, a1, a2, x.data, y.data, out = xbar.data)

        return xbar

    @classmethod
    def hyp2f0(cls, a1, a2, x):
        """ computes y = hyp2f0(a1, a2, x) in UTP arithmetic"""

        retval = x.clone()
        cls._hyp2f0(a1, a2, x.data, out = retval.data)
        return retval

    @classmethod
    def pb_hyp2f0(cls, ybar, a1, a2, x, y, out=None):
        """ computes bar y dy = bar x dx in UTP arithmetic"""
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            # out = (a1bar, a2bar, xbar)
            xbar = out[2]

        cls._pb_hyp2f0(ybar.data, a1, a2, x.data, y.data, out = xbar.data)

        return xbar

    @classmethod
    def hyp0f1(cls, b, x):
        """ computes y = hyp0f1(b, x) in UTP arithmetic"""

        retval = x.clone()
        cls._hyp0f1(b, x.data, out = retval.data)
        return retval

    @classmethod
    def pb_hyp0f1(cls, ybar, b, x, y, out=None):
        """ computes bar y dy = bar x dx in UTP arithmetic"""
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            # out = (bbar, xbar)
            xbar = out[1]

        cls._pb_hyp0f1(ybar.data, b, x.data, y.data, out = xbar.data)

        return xbar

    @classmethod
    def polygamma(cls, n, x):
        """ computes y = polygamma(n, x) in UTP arithmetic"""

        retval = x.clone()
        cls._polygamma(n, x.data, out = retval.data)
        return retval

    @classmethod
    def pb_polygamma(cls, ybar, n, x, y, out=None):
        """ computes bar y dy = bar x dx in UTP arithmetic"""
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            # out = (nbar, xbar)
            xbar = out[1]

        cls._pb_polygamma(ybar.data, n, x.data, y.data, out = xbar.data)

        return xbar

    @classmethod
    def psi(cls, x):
        """ computes y = psi(x) in UTP arithmetic"""

        retval = x.clone()
        cls._psi(x.data, out = retval.data)
        return retval

    @classmethod
    def pb_psi(cls, ybar, x, y, out=None):
        """ computes bar y dy = bar x dx in UTP arithmetic"""
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar, = out

        cls._pb_psi(ybar.data, x.data, y.data, out = xbar.data)
        return xbar

    @classmethod
    def reciprocal(cls, x):
        """ computes y = reciprocal(x) in UTP arithmetic"""

        retval = x.clone()
        cls._reciprocal(x.data, out = retval.data)
        return retval

    @classmethod
    def pb_reciprocal(cls, ybar, x, y, out=None):
        """ computes bar y dy = bar x dx in UTP arithmetic"""
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar, = out

        cls._pb_reciprocal(ybar.data, x.data, y.data, out = xbar.data)
        return xbar

    @classmethod
    def gammaln(cls, x):
        """ computes y = gammaln(x) in UTP arithmetic"""

        retval = x.clone()
        cls._gammaln(x.data, out = retval.data)
        return retval

    @classmethod
    def pb_gammaln(cls, ybar, x, y, out=None):
        """ computes bar y dy = bar x dx in UTP arithmetic"""
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar, = out

        cls._pb_gammaln(ybar.data, x.data, y.data, out = xbar.data)
        return xbar

    @classmethod
    def minimum(cls, x, y):
        # FIXME: this typechecking is probably not flexible enough
        # FIXME: also add pullback
        if isinstance(x, UTPM) and isinstance(y, UTPM):
            return UTPM(cls._minimum(x.data, y.data))
        elif isinstance(x, numpy.ndarray) and isinstance(y, numpy.ndarray):
            return numpy.minimum(x, y)
        else:
            raise NotImplementedError(
                    'this combination of types is not yet implemented')

    @classmethod
    def maximum(cls, x, y):
        # FIXME: this typechecking is probably not flexible enough
        # FIXME: also add pullback
        if isinstance(x, UTPM) and isinstance(y, UTPM):
            return UTPM(cls._maximum(x.data, y.data))
        elif isinstance(x, numpy.ndarray) and isinstance(y, numpy.ndarray):
            return numpy.maximum(x, y)
        else:
            raise NotImplementedError(
                    'this combination of types is not yet implemented')

    @classmethod
    def absolute(cls, x):
        """ computes y = absolute(x) in UTP arithmetic"""

        retval = x.clone()
        cls._absolute(x.data, out = retval.data)
        return retval

    @classmethod
    def pb_absolute(cls, ybar, x, y, out=None):
        """ computes ybar * ydot = xbar * xdot in UTP arithmetic"""

        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar, = out

        cls._pb_absolute(ybar.data, x.data, y.data, out = xbar.data)
        return xbar

    @classmethod
    def negative(cls, x):
        """ computes y = negative(x) in UTP arithmetic"""

        retval = x.clone()
        cls._negative(x.data, out = retval.data)
        return retval

    @classmethod
    def pb_negative(cls, ybar, x, y, out=None):
        """ computes ybar * ydot = xbar * xdot in UTP arithmetic"""

        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar, = out

        cls._pb_negative(ybar.data, x.data, y.data, out = xbar.data)
        return xbar

    @classmethod
    def square(cls, x):
        """ computes y = square(x) in UTP arithmetic"""

        retval = x.clone()
        cls._square(x.data, out = retval.data)
        return retval

    @classmethod
    def pb_square(cls, ybar, x, y, out=None):
        """ computes ybar * ydot = xbar * xdot in UTP arithmetic"""

        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar, = out

        cls._pb_square(ybar.data, x.data, y.data, out = xbar.data)
        return xbar

    @classmethod
    def erf(cls, x):
        """ computes y = erf(x) in UTP arithmetic"""

        retval = x.clone()
        cls._erf(x.data, out = retval.data)
        return retval


    @classmethod
    def pb_erf(cls, ybar, x, y, out=None):
        """ computes ybar * ydot = xbar * xdot in UTP arithmetic"""

        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar, = out

        cls._pb_erf(ybar.data, x.data, y.data, out = xbar.data)
        return xbar


    @classmethod
    def erfi(cls, x):
        """ computes y = erfi(x) in UTP arithmetic"""

        retval = x.clone()
        cls._erfi(x.data, out = retval.data)
        return retval

    @classmethod
    def pb_erfi(cls, ybar, x, y, out=None):
        """ computes ybar * ydot = xbar * xdot in UTP arithmetic"""

        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar, = out

        cls._pb_erfi(ybar.data, x.data, y.data, out = xbar.data)
        return xbar


    @classmethod
    def dawsn(cls, x):
        """ computes y = dawsn(x) in UTP arithmetic"""

        retval = x.clone()
        cls._dawsn(x.data, out = retval.data)
        return retval

    @classmethod
    def pb_dawsn(cls, ybar, x, y, out=None):
        """ computes ybar * ydot = xbar * xdot in UTP arithmetic"""

        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar, = out

        cls._pb_dawsn(ybar.data, x.data, y.data, out = xbar.data)
        return xbar


    @classmethod
    def logit(cls, x):
        """ computes y = logit(x) in UTP arithmetic"""

        retval = x.clone()
        cls._logit(x.data, out = retval.data)
        return retval

    @classmethod
    def pb_logit(cls, ybar, x, y, out=None):
        """ computes ybar * ydot = xbar * xdot in UTP arithmetic"""

        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar, = out

        cls._pb_logit(ybar.data, x.data, y.data, out = xbar.data)
        return xbar

    @classmethod
    def expit(cls, x):
        """ computes y = expit(x) in UTP arithmetic"""

        retval = x.clone()
        cls._expit(x.data, out = retval.data)
        return retval

    @classmethod
    def pb_expit(cls, ybar, x, y, out=None):
        """ computes ybar * ydot = xbar * xdot in UTP arithmetic"""

        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar, = out

        cls._pb_expit(ybar.data, x.data, y.data, out = xbar.data)
        return xbar


    def sum(self, axis=None, dtype=None, out=None):
        if dtype != None or out != None:
            raise NotImplementedError('not implemented yet')

        if axis == None:
            tmp = numpy.prod(self.data.shape[2:])
            return UTPM(numpy.sum(self.data.reshape(self.data.shape[:2] + (tmp,)), axis = 2))
        else:
            if axis < 0:
                a = self.data.ndim + axis
            else:
                a = axis + 2
            return UTPM(numpy.sum(self.data, axis = a))

    @classmethod
    def pb_sum(cls, ybar, x, y, axis, dtype, out2, out = None):

        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar = out[0]

        if axis == None:

            tmp = xbar.data.T
            tmp += ybar.data.T

        else:

            if axis < 0:
                a = x.data.ndim + axis

            else:
                a = axis + 2

            shp = list(x.data.shape)
            shp[a] = 1
            tmp = ybar.data.reshape(shp)
            xbar.data += tmp

        return xbar

    # def prod(self, axis=None, dtype=None, out=None):
    #     if dtype != None or out != None:
    #         raise NotImplementedError('not implemented yet')

    #     if axis == None:
    #         D,P = self.data.shape[:2]
    #         tmp = self.__class__(numpy.zeros((D,P)))
    #         tmp.data[0] = 1.
    #         for xi in self.flat:
    #             tmp *= xi
    #         return tmp
    #     else:
    #         raise NotImplementedError('should implement this case')

    # @classmethod
    # def pb_prod(cls, ybar, x, axis, dtype, dummy, y, out = None):
    #     D,P = x.data.shape[:2]
    #     if out == None:
    #         xbar = x.zeros_like()

    #     else:
    #         xbar = out[0]

    #     tmp = x.copy()

    #     tmp = y/x
    #     xbar += tmp

    #     if numpy.any(numpy.isnan(xbar.data)):
    #         raise NotImplementedError('should treat the case when one element of the product is zero')


    #     return xbar



    @classmethod
    def pb_sincos(cls, sbar, cbar, x, s, c, out = None):
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar, = out

        cls._pb_sincos(sbar.data, cbar.data, x.data, s.data, c.data, out = xbar.data)

        return out

    def arcsin(self):
        """ computes y = arcsin(x) in UTP arithmetic"""
        rety = self.clone()
        retz = self.clone()
        self._arcsin(self.data, out = (rety.data, retz.data))
        return rety

    def arccos(self):
        """ computes y = arccos(x) in UTP arithmetic"""
        rety = self.clone()
        retz = self.clone()
        self._arccos(self.data, out = (rety.data, retz.data))
        return rety

    def arctan(self):
        """ computes y = arctan(x) in UTP arithmetic"""
        rety = self.clone()
        retz = self.clone()
        self._arctan(self.data, out = (rety.data, retz.data))
        return rety


    def sinhcosh(self):
        """ simultaneously computes s = sinh(x) and c = cosh(x) in UTP arithmetic"""
        rets = self.clone()
        retc = self.clone()
        self._sinhcosh(self.data, out = (rets.data, retc.data))
        return rets, retc

    def sinh(self):
        """ computes y = sinh(x) in UTP arithmetic """
        retval = self.clone()
        tmp = self.clone()
        self._sinhcosh(self.data, out = (retval.data, tmp.data))
        return retval

    def cosh(self):
        """ computes y = cosh(x) in UTP arithmetic """
        retval = self.clone()
        tmp = self.clone()
        self._sinhcosh(self.data, out = (tmp.data, retval.data))
        return retval

    def tanh(self):
        """ computes y = tanh(x) in UTP arithmetic """
        retval = self.clone()
        tmp = self.clone()
        self._tanhsech2(self.data, out = (retval.data, tmp.data))
        return retval

    def sign(self):
        """ computes y = sign(x) in UTP arithmetic"""
        retval = self.clone()
        self._sign(self.data, out = retval.data)
        return retval

    @classmethod
    def pb_sign(cls, ybar, x, y, out=None):
        """ computes bar y dy = bar x dx in UTP arithmetic"""
        if out == None:
            D,P = x.data.shape[:2]
            xbar = x.zeros_like()
        else:
            xbar, = out
        cls._pb_sign(ybar.data, x.data, y.data, out = xbar.data)
        return out


    def __abs__(self):
        """ absolute value of polynomials

        FIXME: theory tells us to check first coefficient if the zero'th coefficient is zero
        """
        # check if zero order coeff is smaller than 0
        tmp = self.data[0] < 0
        retval = self.clone()
        retval.data *= (-1)**tmp

        return retval

    def fabs(self):
        return self.__abs__()

    def __neg__(self):
        return self.__class__.neg(self)

    def __lt__(self, other):
        if isinstance(other,self.__class__):
            return numpy.all(self.data[0,...] < other.data[0,...])
        else:
            return numpy.all(self.data[0,...] < other)

    def __le__(self, other):
        if isinstance(other,self.__class__):
            return numpy.all(self.data[0,...] <= other.data[0,...])
        else:
            return numpy.all(self.data[0,...] <= other)

    def __gt__(self, other):
        if isinstance(other,self.__class__):
            return numpy.all(self.data[0,...] > other.data[0,...])
        else:
            return numpy.all(self.data[0,...] > other)

    def __ge__(self, other):
        if isinstance(other,self.__class__):
            return numpy.all(self.data[0,...] >= other.data[0,...])
        else:
            return numpy.all(self.data[0,...] >= other)

    def __eq__(self, other):
        if isinstance(other,self.__class__):
            return numpy.all(self.data[0,...] == other.data[0,...])
        else:
            return numpy.all(self.data[0,...] == other)

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

    @classmethod
    def det(cls, x):
        """ returns a new UTPM in standard format, i.e. the matrices are 1x1 matrices"""
        D,P = x.data.shape[:2]
        L = cls.cholesky(x)
        return numpy.prod(cls.diag(L))**2

    @classmethod
    def pb_det(cls, ybar, x, y, out = None):
        if out == None:
            out = (x.zeros_like(),)

        raise NotImplementedError('should implement that')

        # xbar, = out
        # Nx = xbar.shape[0]
        # for nx in range(Nx):
        #     xbar[nx,nx] += ybar

        # return xbar




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

    def __len__(self):
        return self.shape[0]

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

    def ones_like(self):
        data = numpy.zeros_like(self.data)
        data[0,...] = 1.
        return self.__class__(data)

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
        return 'UTPM(' + self.__str__() + ')'

    @classmethod
    def pb_zeros(cls, *args, **kwargs):
        pass

    @classmethod
    def tril(cls, x, out = None):
        out = x.zeros_like()
        D,P = out.data.shape[:2]
        # print D,P
        for d in range(D):
            for p in range(P):
                out.data[d,p] = numpy.tril(x.data[d,p])

        return out

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
    def init_jacobian(cls, x, dtype=None):
        """ initializes this UTPM instance to compute the Jacobian,

        it is possible to force the dtype to a certain dtype,
        if no dtype is provided, the dtype is inferred from x
        """

        x = numpy.asarray(x)

        if dtype==None:
            # try to infer the dtype from x
            dtype= x.dtype

            if dtype==int:
                dtype=float


        shp = numpy.shape(x)
        data = numpy.zeros(numpy.hstack( [2, numpy.size(x), shp]), dtype=dtype)
        data[0] = x
        data[1,:].flat = numpy.eye(numpy.size(x))

        return cls(data)

    @classmethod
    def extract_jacobian(cls, x):
        """ extracts the Jacobian from a UTPM instance
        if x.ndim == 1 it is equivalent to the gradient
        """
        return x.data[1,...].transpose([i for i in range(1,x.data[1,...].ndim)] + [0])

    @classmethod
    def init_jac_vec(cls, x, v, dtype=None):
        """ initializes this UTPM instance to compute the Jacobian vector product J v,

        it is possible to force the dtype to a certain dtype,
        if no dtype is provided, the dtype is inferred from x
        """

        x = numpy.asarray(x)

        if dtype==None:
            # try to infer the dtype from x
            dtype= x.dtype

            if dtype==int:
                dtype=float


        shp = numpy.shape(x)
        data = numpy.zeros(numpy.hstack( [2, 1, shp]), dtype=dtype)
        data[0,0] = x
        data[1,0] = v
        return cls(data)

    @classmethod
    def extract_jac_vec(cls, x):
        """ extracts the Jacobian vector from a UTPM instance
        if x.ndim == 1 it is equivalent to the gradient
        """
        return x.data[1,...].transpose([i for i in range(1,x.data[1,...].ndim)] + [0])[:,0]


    @classmethod
    def init_tensor(cls, d, x):
        """ initializes this UTPM instance to compute the dth degree derivative tensor,
        e.g. d=2 is the Hessian
        """

        import algopy.exact_interpolation as exint
        x = numpy.asarray(x)

        if x.ndim != 1:
            raise NotImplementedError('non vector inputs are not implemented yet')

        N = numpy.size(x)
        Gamma, rays = exint.generate_Gamma_and_rays(N,d)

        data = numpy.zeros(numpy.hstack([d+1,rays.shape]))
        data[0] = x
        data[1] = rays
        return cls(data)

    @classmethod
    def extract_tensor(cls, N, y, as_full_matrix = True):
        """ extracts the Hessian of shape (N,N) from the UTPM instance y
        """

        import algopy.exact_interpolation as exint
        d = y.data.shape[0]-1
        Gamma, rays = exint.generate_Gamma_and_rays(N,d)
        tmp = numpy.dot(Gamma,y.data[d])

        if as_full_matrix == False:
            return tmp

        else:
            retval = numpy.zeros((N,N))
            mi = exint.generate_multi_indices(N,d)
            pos = exint.convert_multi_indices_to_pos(mi)

            for ni in range(mi.shape[0]):
                # print 'ni=',ni, mi[ni], pos[ni], tmp[ni]
                for perm in exint.generate_permutations(list(pos[ni])):
                    retval[perm[0],perm[1]] = tmp[ni]*numpy.max(mi[ni])

            return retval


    @classmethod
    def init_hessian(cls, x):
        """ initializes this UTPM instance to compute the Hessian
        """

        x = numpy.ravel(x)

        # generate directions
        N = x.size
        M = (N*(N+1))/2
        L = (N*(N-1))/2
        S = numpy.zeros((N,M), dtype=x.dtype)

        s = 0
        i = 0
        for n in range(1,N+1):
            S[-n:,s:s+n] = numpy.eye(n)
            S[-n,s:s+n] = numpy.ones(n)
            s+=n
            i+=1
        S = S[::-1].T

        data = numpy.zeros(numpy.hstack([3,S.shape]), dtype=x.dtype)
        data[0] = x
        data[1] = S
        return cls(data)

    @classmethod
    def extract_hessian(cls, N, y, as_full_matrix = True, use_mpmath=False):
        """ extracts the Hessian of shape (N,N) from the UTPM instance y
        """

        if use_mpmath:
            import mpmath
            mpmath.dps = 50


        H = numpy.zeros((N,N),dtype=y.data.dtype)
        for n in range(N):
            for m in range(n):
                a =  sum(range(n+1))
                b =  sum(range(m+1))
                k =  sum(range(n+2)) - m - 1
                #print 'k,a,b=', k,a,b
                if n!=m:

                    if use_mpmath:
                        tmp = (mpmath.mpf(y.data[2,k]) - mpmath.mpf(y.data[2,a]) - mpmath.mpf(y.data[2,b]))
                    else:
                        tmp = (y.data[2,k] - y.data[2,a] - y.data[2,b])

                    H[m,n]= H[n,m]= tmp
            a =  sum(range(n+1))
            H[n,n] = 2*y.data[2,a]
        return H




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
    def outer(cls, x, y, out = None):
        """
        out = outer(x,y)
        """

        if isinstance(x, UTPM) and isinstance(y, UTPM):
            x_shp = x.data.shape
            y_shp = y.data.shape

            assert x_shp[:2] == y_shp[:2]
            assert len(y_shp[2:]) == 1

            out_shp = x_shp + x_shp[-1:]
            out = cls(cls.__zeros__(out_shp, dtype = x.data.dtype))
            cls._outer( x.data, y.data, out = out.data)

        elif isinstance(x, UTPM) and isinstance(y, numpy.ndarray):
            x_shp = x.data.shape
            out_shp = x_shp + x_shp[-1:]
            out = cls(cls.__zeros__(out_shp, dtype = x.data.dtype))
            cls._outer_non_utpm_y( x.data, y, out = out.data)

        elif isinstance(x, numpy.ndarray) and isinstance(y, UTPM):
            y_shp = y.data.shape
            out_shp = y_shp + y_shp[-1:]
            out = cls(cls.__zeros__(out_shp, dtype = y.data.dtype))
            cls._outer_non_utpm_x( x, y.data, out = out.data)

        else:
            raise NotImplementedError('this operation is not supported')

        return out

    @classmethod
    def pb_outer(cls, zbar, x, y, z, out = None):
        if out == None:
            D,P = y.data.shape[:2]
            xbar = x.zeros_like()
            ybar = y.zeros_like()

        else:
            xbar, ybar = out

        cls._outer_pullback(zbar.data, x.data, y.data, z.data, out = (xbar.data, ybar.data))
        return (xbar,ybar)


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
                print(ValueError('A.data.shape = %s does not match x.data.shape = %s'%(str(A_shp), str(x_shp))))

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
            Abar, = out

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
    def pb___truediv__(cls, zbar, x, y , z, out = None):
        return cls.pb_truediv(zbar, x, y , z, out = out)

    @classmethod
    def pb_add(cls, zbar, x, y, z, out=None):
        if out == None:
            D, P = y.data.shape[:2]
            xbar = x.zeros_like()
            ybar = y.zeros_like()

        else:
            xbar, ybar = out

        if isinstance(xbar, UTPM):
            xbar2, zbar2 = cls.broadcast(xbar, zbar)

            # print 'xbar = ', xbar
            # print 'zbar = ', zbar
            # print 'xbar2 = ', xbar2
            # print 'zbar2 = ', zbar2

            # print 'xbar2.data.strides = ', xbar2.data.strides
            # print 'zbar2.data.strides = ', zbar2.data.strides
            # print 'xbar2 + zbar2 = ', xbar2 + zbar2

            workaround_strides_function(xbar2, zbar2, operator.iadd)
            # xbar2[...] = xbar2 + zbar2

            # print 'after update'
            # print 'xbar2 =\n', xbar2
            # print 'xbar =\n', xbar

        if isinstance(ybar, UTPM):
            ybar2, zbar2 = cls.broadcast(ybar, zbar)
            workaround_strides_function(ybar2, zbar2, operator.iadd)
            # ybar2 += zbar2
        # print 'ybar2.data.shape=',ybar2.data.shape


        return (xbar, ybar)


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

        if isinstance(x, UTPM):
            xbar2,zbar2 = cls.broadcast(xbar, zbar)
            workaround_strides_function(xbar2, zbar2, operator.iadd)
            # xbar2 += zbar2

        if isinstance(y, UTPM):
            ybar2,zbar2 = cls.broadcast(ybar, zbar)
            workaround_strides_function(ybar2, zbar2, operator.isub)
            # ybar2 -= zbar2

        return (xbar,ybar)


    @classmethod
    def pb_mul(cls, zbar, x, y , z, out = None):

        if isinstance(x, UTPM) and isinstance(y, UTPM):
            if out == None:
                D,P = z.data.shape[:2]
                xbar = x.zeros_like()
                ybar = y.zeros_like()

            else:
                xbar, ybar = out

            xbar2, tmp = cls.broadcast(xbar, zbar)
            ybar2, tmp = cls.broadcast(ybar, zbar)

            # xbar2 += zbar * y
            workaround_strides_function(xbar2, zbar * y, operator.iadd)
            # ybar2 += zbar * x
            workaround_strides_function(ybar2, zbar * x, operator.iadd)

            return (xbar, ybar)

        elif isinstance(x, UTPM):
            if out == None:
                D, P = z.data.shape[:2]
                xbar = x.zeros_like()
                ybar = None

            else:
                xbar, ybar = out

            xbar2, tmp = cls.broadcast(xbar, zbar)

            workaround_strides_function(xbar2, zbar * y, operator.iadd)
            # xbar2 += zbar * y

            return (xbar, ybar)

        elif isinstance(y, UTPM):
            if out == None:
                D, P = z.data.shape[:2]
                xbar = None
                ybar = y.zeros_like()

            else:
                xbar, ybar = out

            ybar2, tmp = cls.broadcast(ybar, zbar)

            workaround_strides_function(xbar2, zbar * x, operator.iadd)
            # ybar2 += zbar * x

            return (xbar, ybar)

        else:
            raise NotImplementedError('not implemented')

    @classmethod
    def pb_truediv(cls, zbar, x, y, z, out=None):

        if isinstance(x, UTPM) and isinstance(y, UTPM):

            if out == None:
                D,P = y.data.shape[:2]
                xbar = x.zeros_like()
                ybar = y.zeros_like()

            else:
                xbar, ybar = out

            x2, y2 = cls.broadcast(x, y)

            xbar2, tmp = cls.broadcast(xbar, zbar)
            ybar2, tmp = cls.broadcast(ybar, zbar)

            tmp = zbar.clone()
            # tmp /= y2
            workaround_strides_function(tmp, y2, operator.itruediv)
            # xbar2 += tmp
            workaround_strides_function(xbar2, tmp, operator.iadd)
            # tmp *= z
            workaround_strides_function(tmp, z, operator.imul)
            # ybar2 -= tmp
            workaround_strides_function(ybar2, tmp, operator.isub)

            return (xbar, ybar)

        elif isinstance(x, UTPM):

            if out == None:
                D, P = z.data.shape[:2]
                xbar = x.zeros_like()
                ybar = None

            else:
                xbar, ybar = out

            xbar2, tmp = cls.broadcast(xbar, zbar)

            # tmp /= y2
            # xbar2 += tmp
            workaround_strides_function(xbar2, zbar / y, operator.iadd)

            return (xbar, ybar)

        elif isinstance(y, UTPM):

            if out == None:
                D, P = z.data.shape[:2]
                xbar = None
                ybar = y.zeros_like()

            else:
                xbar, ybar = out

            ybar2, tmp = cls.broadcast(ybar, zbar)
            workaround_strides_function(ybar2, zbar / y * z, operator.isub)

            return (xbar, ybar)

    @classmethod
    def broadcast(cls, x,y):
        """
        this is the UTPM equivalent to numpy.broadcast_arrays
        """
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
    def pb_dot(cls, zbar, x, y, z, out = None):
        if out == None:
            D,P = y.data.shape[:2]
            xbar = x.zeros_like()
            ybar = y.zeros_like()

        else:
            xbar, ybar = out

        # print 'x = ', type(x)
        # print 'y = ',type(y)
        # print 'z = ',type(z)

        # print 'xbar = ', type(xbar)
        # print 'ybar = ',type(ybar)
        # print 'zbar = ',type(zbar)

        if not isinstance(x,cls):
            D,P = z.data.shape[:2]
            tmp = cls(numpy.zeros((D,P) + x.shape,dtype=z.data.dtype))
            tmp[...] = x[...]
            x = tmp

        if not isinstance(xbar,cls):
            xbar = cls(numpy.zeros((D,P) + x.shape,dtype=z.data.dtype))

        if not isinstance(y,cls):
            D,P = xbar.data.shape[:2]
            tmp = cls(numpy.zeros((D,P) + y.shape,dtype=z.data.dtype))
            tmp[...] = y[...]
            y = tmp

        if not isinstance(ybar,cls):
            ybar = cls(numpy.zeros((D,P) + y.shape,dtype=z.data.dtype))

        cls._dot_pullback(zbar.data, x.data, y.data, z.data, out = (xbar.data, ybar.data))
        return (xbar,ybar)

    @classmethod
    def pb_reshape(cls, ybar, x, newshape, y, out = None):
        if out == None:
            D,P = y.data.shape[:2]
            xbar = x.zeros_like()

        else:
            xbar = out[0]

        cls._pb_reshape(ybar.data, x.data, y.data, out = xbar.data)
        return xbar


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
    def qr(cls, A, out = None, work = None, epsilon = 1e-14):
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
    def eigh(cls, A, out = None, epsilon = 1e-8):
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
    def eigh1(cls, A, out = None, epsilon = 1e-8):
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
    def symvec(cls, A, UPLO = 'F'):
        """
        maps a symmetric matrix to a vector containing the distinct elements
        """
        import algopy.utils
        return algopy.utils.symvec(A, UPLO=UPLO)

    @classmethod
    def pb_symvec(cls, vbar, A, UPLO, v, out = None):

        if out == None:
            Abar = A.zeros_like()

        else:
            Abar = out[0]

        N,M = A.shape

        if UPLO=='F':
            count = 0
            for row in range(N):
                for col in range(row,N):
                    Abar[row,col] += 0.5 * vbar[count]
                    Abar[col,row] += 0.5 * vbar[count]
                    count +=1

        elif UPLO=='L':
            count = 0
            for n in range(N):
                for m in range(n,N):
                    Abar[m,n] = vbar[count]
                    count +=1

        elif UPLO=='U':
            count = 0
            for n in range(N):
                for m in range(n,N):
                    Abar[n,m] = vbar[count]
                    count +=1

        else:
            err_str = "UPLO must be either 'F','L', or 'U'\n"
            err_str+= "however, provided UPLO=%s"%UPLO
            raise ValueError(err_str)

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
        if abs(int(tmp) - tmp) > 1e-16:
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

        N = A.shape[0]

        count = 0
        for row in range(N):
            vbar[count] += Abar[row,row]
            count += 1
            for col in range(row+1,N):
                vbar[count] += Abar[col,row]
                vbar[count] += Abar[row,col]
                count +=1

        return vbar



    @classmethod
    def iouter(cls, x, y, out):
        cls._iouter(x.data, y.data, out.data)
        return out

    def reshape(self,  newshape, order = 'C'):
        if order != 'C':
            raise NotImplementedError('should implement that')
        cls = self.__class__
        return cls(cls._reshape(self.data, newshape, order = order))


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


class UTP(UTPM):
    """
    UTP(X, vectorized=False)

    Univariate Taylor Polynomial (UTP)
    with coefficients that are arbitrary numpy.ndarrays

    Attributes
    ----------
    data: numpy.ndarray
        underlying datastructure, a numpy.array of shape (D,P) + UTP.shape

    coeff: numpy.ndarray like structure
        is accessed just like UTP.data but has the shape (D,) + UTP.shape if
        vectorized=False and exactly the same as UTP.data when vectorized=True

    vectorized: bool
        whether the UTP is vectorized or not (default is False)

    All other attributes are motivated from numpy.ndarray and return
    size, shape, ndim of an individual coefficient of the UTP. E.g.,

    T: UTP
        Transpose of the UTP
    size: int
        Number of elements in a UTP coefficient
    shape: tuple of ints
        Shape of a UTP coefficient
    ndim: int
        The number of dimensions of a UTP coefficient

    Parameters
    ----------

    X: numpy.ndarray with shape (D, P, N1, N2, N3, ...) if vectorized=True
       otherwise a (D, N1, N2, N3, ...) array


    Remark:
        This class provides an improved userinterface compared to the class UTPM.

        The difference is mainly the initialization.

        E.g.::

            x = UTP([1,2,3])

        is equivalent to::

            x = UTP([1,2,3], P=1)
            x = UTPM([[1],[2],[3]])

        and::
            x = UTP([[1,2],[2,3],[3,4]], P=2)

        is equivalent to::

            x = UTPM([[1,2],[2,3],[3,4]])
    """

    def __init__(self, X, vectorized=False):
        """
        see self.__class__.__doc__ for information
        """
        Ndim = numpy.ndim(X)
        self.vectorized = vectorized
        if Ndim >= 1:
            self.data = numpy.asarray(X)
            if vectorized == False:
                shp = self.data.shape
                self.data = self.data.reshape(shp[:1] + (1,) + shp[1:])
        else:
            raise NotImplementedError

    @property
    def coeff(self, *args, **kwargs):
        if self.vectorized == False:
            return self.data[:,0,...]
        else:
            return self.data

    def __str__(self):
        """ return string representation """
        return str(self.coeff)
