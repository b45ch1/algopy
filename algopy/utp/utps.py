"""
Implementation of the truncated univariate polynomials
The algebraic class is

R[t]/<t^D>

where R is the field of real numbers

"""

import numpy


class UTPS:
    """
    UTPS == Univariate Taylor Polynomial on Scalars
    This class implements univariate Taylor arithmetic on real numbers, i.e.
    [x] = \sum_{d=0}^D x_d t^d
    x_d = \frac{d^d}{dt^d}|_{t=0} \sum_{c=0}^D x_c t^c

    The Taylor Coefficients are stored in `self.tc`.

    in vector forward mode
    Input:
    in the most general form, the input is a 4-tensor.
    We use the notation:
    P: number of directions
    D: degree of the Taylor series

    shape([x]) = (D,P)
    """

    def __init__(self,  *taylor_coeffs):
        """Constructor takes a list, array, tuple and variable lenght input"""
        if not numpy.isscalar(taylor_coeffs[0]):
            taylor_coeffs = numpy.array(taylor_coeffs[0],dtype=float)
        self.tc = numpy.array(taylor_coeffs,dtype=float)
        self.off = 0
        self.shp = numpy.shape(self.tc)
        if len(self.shp) ==2:
            self.Ndir = numpy.shape(self.tc)[1]
            self.D = numpy.shape(self.tc)[0]
        else:
            self.D = numpy.shape(self.tc)[0]


    def get_tc(self):
        return self.tc
    def set_tc(self,x):
        self.tc[:] = x[:]
    tc = property(get_tc, set_tc)

    def copy(self):
        return UTPS(self.tc)

    def __add__(self, rhs):
        """compute new Taylorseries of the function f(x,y) = x+y, where x and y UTPS objects"""
        if isinstance(rhs, UTPS):
            return UTPS(self.tc + rhs.tc)
        elif numpy.isscalar(rhs):
            retval = UTPS(numpy.copy(self.tc))
            retval.tc[0] += rhs
            return retval
        else:
            raise NotImplementedError

    def __radd__(self, val):
        return self+val

    def __sub__(self, rhs):
        if isinstance(rhs, UTPS):
            return UTPS(self.tc - rhs.tc)

        elif numpy.isscalar(rhs):
            retval = UTPS(numpy.copy(self.tc))
            retval.tc[0] -= rhs
            return retval
        else:
            raise NotImplementedError

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, rhs):
        """compute new Taylorseries of the function f(x,y) = x*y, where x and y UTPS objects"""
        if isinstance(rhs, UTPS):
            return UTPS(numpy.array(
                    [ numpy.sum(self.tc[:k+1] * rhs.tc[k::-1], axis = 0) for k in range(self.D)]
                    ))
        elif numpy.isscalar(rhs):
            return UTPS(rhs * self.tc)
        else:
            raise NotImplementedError("%s multiplication with UTPS object" % type(rhs))

    def __rmul__(self, val):
        return self*val

    def __div__(self, rhs):
        """compute new Taylorseries of the function f(x,y) = x/y, where x and y UTPS objects"""
        if isinstance(rhs, UTPS):
            y = UTPS(numpy.zeros(self.shp))
            for k in range(self.D):
                y.tc[k] = 1./ rhs.tc[0] * ( self.tc[k] - numpy.sum(y.tc[:k] * rhs.tc[k:0:-1], axis = 0))
            return y
        else:
            y = UTPS(numpy.zeros(self.shp))
            for k in range(self.D):
                y.tc[k] =  self.tc[k]/rhs
            return y


    def __rdiv__(self, val):
        tmp = numpy.zeros(self.shp)
        tmp[0] = val
        return UTPS(tmp)/self

    def __neg__(self):
        return UTPS(-self.tc)


    # TODO: the comparisons check only if any x_0 satisfies the comparison!!
    def __lt__(self,rhs):
        if numpy.isscalar(rhs):
            return (self.tc[0] < rhs).any()
        return (self.tc[0] < rhs.tc[0]).any()

    def __le__(self,rhs):
        if numpy.isscalar(rhs):
            return (self.tc[0] <= rhs).any()
        return (self.tc[0] <= rhs.tc[0]).any()

    def __eq__(self,rhs):
        if numpy.isscalar(rhs):
            return (self.tc[0] == rhs).any()
        return (self.tc[0] == rhs.tc[0]).any()

    def __ne__(self,rhs):
        if numpy.isscalar(rhs):
            return (self.tc[0] != rhs).any()
        return (self.tc[0] != rhs.tc[0]).any()

    def __ge__(self,rhs):
        if numpy.isscalar(rhs):
            return (self.tc[0] >= rhs).any()
        return (self.tc[0] >= rhs.tc[0]).any()

    def __gt__(self,rhs):
        if numpy.isscalar(rhs):
            return (self.tc[0] > rhs).any()
        return (self.tc[0] > rhs.tc[0]).any()

    def __abs__(self):
        tmp = self.copy()
        if (tmp.tc[0] < 0).any():
            tmp.tc[:] = -tmp.tc[:]
        elif (tmp.tc[0] == 0).any():
            print("UTPS with abs(x) at x=0")
        return tmp

    def __pow__(self, exponent):
        """Computes the power: x^n, where n must be an int"""
        if isinstance(exponent, int):
            tmp = 1
            for i in range(exponent):
                tmp=tmp*self
            return tmp
        else:
            raise TypeError("Second argumnet must be an integer")

    def sqrt(self):
        y = UTPS(numpy.zeros(self.shp))
        y.tc[0] = numpy.sqrt(self.tc[0])
        for k in range(1,self.D):
            y.tc[k] = 1./(2*y.tc[0]) * ( self.tc[k] - numpy.sum( y.tc[1:k] * y.tc[k-1:0:-1]))
        return y
            
    def exp(self):
        y = self.zeros_like()
        y.tc[0] = numpy.exp(self.tc[0])
        factor  = numpy.arange(1, self.D, dtype=float)
        xtctilde = factor * self.tc[1:]
        for d in range(1, self.D):
            y.tc[d] = numpy.sum(y.tc[:d][::-1]*xtctilde[:d])/d
        return y

    def zeros_like(self):
        return UTPS(numpy.zeros_like(self.tc))
    
    def __str__(self):
        """human readable representation of the UTPS object for printing >>print UTPS([1,2,3]) """
        return 'a(%s)'%str(self.tc)

    def __repr__(self):
        """ human readable output of the UTPS object or debugging UTPS([1,2,3]).__repr__()"""
        return 'UTPS object with taylor coefficients %s'%self.tc