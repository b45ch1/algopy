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
    in the most general form, the input is a 2-tensor.
    We use the notation:
    P: number of directions
    D: degree of the Taylor series

    shape([x]) = (D,P)
    """

    def __init__(self,  taylor_coeffs):
        """Constructor takes a list, array, tuple"""
        self.tc = numpy.asarray(taylor_coeffs, dtype=float)
        self.off = 0
        if numpy.ndim(self.tc) == 1:
            self.tc = numpy.reshape(self.tc, (self.tc.shape[0],1))
        self.shp = self.tc.shape
        self.D, self.P = self.shp
        
        self.data = self.tc

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
        return str(self)
        
        
        
def qr(in_A):
    """
    QR decomposition of A
   
    Q,R = qr(A)
    
    """
    # input checks
    Ndim = numpy.ndim(in_A)
    assert Ndim == 2
    N,M = numpy.shape(in_A)
    assert N==M
    D,P = in_A[0,0].data.shape

    # prepare R and QT
    R = in_A.copy()
    QT = numpy.array([[UTPS(numpy.zeros((D,P))) for c in range(N)] for r in range(N) ])
    for n in range(N):
        QT[n,n].data[0,:] = 1

    # main algorithm
    for n in range(N):
        for m in range(n+1,N):
            a = R[n,n]
            b = R[m,n]
            r = numpy.sqrt(a**2 + b**2)
            c = a/r
            s = b/r

            for k in range(N):
                Rnk = R[n,k]
    
                R[n,k] = c*Rnk + s*R[m,k]
                R[m,k] =-s*Rnk + c*R[m,k];

                QTnk = QT[n,k]
                QT[n,k] = c*QTnk + s*QT[m,k]
                QT[m,k] =-s*QTnk + c*QT[m,k];
            # #print 'QT:\n',QT
            # #print 'R:\n',R
            # #print '-------------'

    return QT.T,R
            
def inv(in_A):
    """
    computes the inverse of A by
    
    STEP 1: QR decomposition
    STEP 2: Solution of the  extended linear system::
    
            (Q R | I) = ( R | QT )
            
            i.e.
            /R_11 R_12 R_13 ... R_1M | 1 0 0 0 ... 0 \
            | 0   R_22 R_23 ... R_2M | 0 1 0 0 ... 0 |
            | 0         ... ... .... |               |
            \                   R_NM | 0 0 0 0 ... 1 /
    
    
    """
    Q,R = qr(in_A)
    QT = Q.T
    N = shape(in_A)[0]
  
    for n in range(N-1,-1,-1):
        Rnn = R[n,n]
        R[n,:] /= Rnn
        QT[n,:] /= Rnn
        for m in range(n+1,N):
            Rnm = R[n,m]
            R[n,m] = 0
            QT[n,:] -= QT[m,:]*Rnm

    return QT