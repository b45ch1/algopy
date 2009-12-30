"""
Implementation of the univariate matrix polynomial.
The algebraic class is

M[t]/<t^D>

where M is the ring of matrices and t an external parameter

"""

import numpy.linalg
import numpy

from algopy.base_type import GradedRing


# override numpy definitions
def shape(x):
    if isinstance(x, UTPM):
        return x.shape
    else:
        return numpy.shape(x)
        
def size(x):
    if isinstance(x, UTPM):
        return x.size
    else:
        return numpy.size(x)
        
def trace(x):
    if isinstance(x, UTPM):
        return x.trace()
    else:
        return numpy.trace(x)
        
def inv(x):
    if isinstance(x, UTPM):
        return UTPM.inv(x)
    else:
        return numpy.linalg.inv(x)
        
def dot(x,y, out = None):
    
    if out != None:
        raise NotImplementedError('should implement that...')
    
    if isinstance(x, UTPM) or isinstance(y, UTPM):
        return UTPM.dot(x,y)
        
    else:
        return numpy.dot(x,y)
        
        
def solve(A,x):
    if isinstance(x, UTPM):
        raise NotImplementedError('should implement that...')
    
    elif isinstance(A, UTPM):
        raise NotImplementedError('should implement that...')
    
    else:
        return numpy.linalg.solve(A,x)
        
def eig(A):
    if isinstance(A, UTPM):
        return UTPM.eig(A)
    
    else:
        return numpy.linalg.eig(A)

def vdot(x,y, z = None):
    """
    vectorized dot
    
    z = vdot(x,y)
    
    Rationale:
        
        given two iteratable containers (list,array,...) x and y
        this function computes::
        
            z[i] = numpy.dot(x[i],y[i])
            
        if z == None, this function allocates the necessary memory
    
    Warning: the naming is inconsistent with numpy.vdot
    Warning: this is a preliminary version that is likely to be changed
    """
    x_shp = numpy.shape(x)
    y_shp = numpy.shape(y)

    if x_shp[-1] != y_shp[-2]:
        raise ValueError('got x.shape = %s and y.shape = %s'%(str(x_shp),str(y_shp)))

    if numpy.ndim(x) == 3:
        P,N,M  = x_shp
        P,M,K  = y_shp
        retval = numpy.zeros((P,N,K))
        for p in range(P):
            retval[p,:,:] = numpy.dot(x[p,:,:], y[p,:,:])

        return retval

    elif numpy.ndim(x) == 4:
        D,P,N,M  = x_shp
        D,P,M,K  = y_shp
        retval = numpy.zeros((D,P,N,K))
        for d in range(D):
            for p in range(P):
                retval[d,p,:,:] = numpy.dot(x[d,p,:,:], y[d,p,:,:])

        return retval

def truncated_triple_dot(X,Y,Z, D):
    """
    computes d^D/dt^D ( [X]_D [Y]_D [Z]_D) with t set to zero after differentiation
    
    X,Y,Z are (DT,P,N,M) arrays s.t. the dimensions match to compute dot(X[d,p,:,:], dot(Y[d,p,:,:], Z[d,p,:,:])) 
    
    """
    import algopy.utp.exact_interpolation
    DT,P,NX,MX = X.shape
    DT,P,NZ,MZ = Z.shape

    multi_indices = algopy.utp.exact_interpolation.generate_multi_indices(3,D)
    retval = numpy.zeros((P,NX,MZ))
    
    for mi in multi_indices:
        for p in range(P):
            if mi[0] == D or mi[1] == D or mi[2] == D:
                continue
            retval[p] += numpy.dot(X[mi[0],p,:,:], numpy.dot(Y[mi[1],p,:,:], Z[mi[2],p,:,:]))
                
    return retval


def combine_blocks(in_X):
    """
    expects an array or list consisting of entries of type UTPM, e.g.
    in_X = [[UTPM1,UTPM2],[UTPM3,UTPM4]]
    and returns
    UTPM([[UTPM1.tc,UTPM2.tc],[UTPM3.tc,UTPM4.tc]])

    """

    in_X = numpy.array(in_X)
    Rb,Cb = numpy.shape(in_X)

    # find the degree D and number of directions P
    D = 0; 	P = 0;

    for r in range(Rb):
        for c in range(Cb):
            D = max(D, in_X[r,c].tc.shape[0])
            P = max(P, in_X[r,c].tc.shape[1])

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
            tc[:,:,rowsums[r]:rowsums[r+1], colsums[c]:colsums[c+1]] = in_X[r,c].tc[:,:,:,:]

    return UTPM(tc)

class UTPM(GradedRing):
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
    See for example __mul__: there, operations of self.tc[:d+1,:,:,:]* rhs.tc[d::-1,:,:,:] has to be performed. One can see, that contiguous memory blocks are used for such operations.

    A disadvantage of this arrangement is: it seems unnatural. It is easier to regard each direction separately.
    """
    
    def __init__(self, X, Xdot = None):
        """ INPUT:	shape([X]) = (D,P,N,M)
        """
        Ndim = numpy.ndim(X)
        if Ndim >= 2:
            self.tc = numpy.asarray(X)
            self.data = self.tc
        else:
            raise NotImplementedError
            
    def __getitem__(self, sl):
        if type(sl) == int or sl == Ellipsis:
            sl = (sl,)
        tmp = self.tc.__getitem__((slice(None),slice(None)) + sl)
        return UTPM(tmp)
        
    def __setitem__(self, sl, rhs):
        if isinstance(rhs, UTPM):
            if type(sl) == int or sl == Ellipsis:
                sl = (sl,)
            return self.tc.__setitem__((slice(None),slice(None)) + sl, rhs.tc)
        else:
            raise NotImplementedError('rhs must be of the type algopy.UTPM!')
        
    def __add__(self,rhs):
        if numpy.isscalar(rhs) or isinstance(rhs,numpy.ndarray):
            retval = UTPM(numpy.copy(self.tc))
            retval.tc[0,:] += rhs
            return retval
        else:
            return UTPM(self.tc + rhs.tc)

    def __sub__(self,rhs):
        if numpy.isscalar(rhs) or isinstance(rhs,numpy.ndarray):
            retval = UTPM(numpy.copy(self.tc))
            retval.tc[0,:] -= rhs
            return retval
        else:
            return UTPM(self.tc - rhs.tc)
            

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
        tmp.tc[0,:,:,:] = rhs
        return tmp/self
        
    def __iadd__(self,rhs):
        if numpy.isscalar(rhs) or isinstance(rhs,numpy.ndarray):
            self.tc[0,...] += rhs
        else:
            self.tc[...] += rhs.tc[...]
        return self
        
    def __isub__(self,rhs):
        if numpy.isscalar(rhs) or isinstance(rhs,numpy.ndarray):
            self.tc[0,...] -= rhs
        else:
            self.tc[...] -= rhs.tc[...]
        return self
        
    def __imul__(self,rhs):
        (D,P) = self.tc.shape[:2]
        if numpy.isscalar(rhs) or isinstance(rhs,numpy.ndarray):
            for d in range(D):
                for p in range(P):
                    self.tc[d,p,...] *= rhs
        else:
            for d in range(D)[::-1]:
                for p in range(P):
                    self.tc[d,p,...] *= rhs.tc[0,p,...]
                    for c in range(d):
                        self.tc[d,p,...] += self.tc[c,p,...] * rhs.tc[d-c,p,...]
        return self
        
    def __idiv__(self,rhs):
        (D,P) = self.tc.shape[:2]
        if numpy.isscalar(rhs) or isinstance(rhs,numpy.ndarray):
            self.tc[...] /= rhs
        else:
            retval = self.clone()
            for d in range(D):
                retval.tc[d,:,...] = 1./ rhs.tc[0,:,...] * ( self.tc[d,:,...] - numpy.sum(retval.tc[:d,:,...] * rhs.tc[d:0:-1,:,...], axis=0))
            self.tc[...] = retval.tc[...]
        return self


    def __neg__(self):
        return UTPM(-self.tc)


    def max(self, axis = None, out = None):
        if out != None:
            raise NotImplementedError('should implement that')

        if axis != None:
            raise NotImplementedError('should implement that')
        
        self_shp = self.data.shape 
        retval_shp = self_shp[:2]
        retval = self.__class__(self.__class__.__zeros__(retval_shp))
        self.__class__._max( self.data, axis = axis, out = retval.data)
        return retval

    def trace(self):
        """ returns a new UTPM in standard format, i.e. the matrices are 1x1 matrices"""
        D,P = self.tc.shape[:2]
        retval = numpy.zeros((D,P))
        for d in range(D):
            for p in range(P):
                retval[d,p] = numpy.trace(self.tc[d,p,...])
        return UTPM(retval)
        
    def FtoJT(self):
        """
        Combines several directional derivatives and combines them to a transposed Jacobian JT, i.e.
        x.tc.shape = (D,P,shp)
        y = x.FtoJT()
        y.tc.shape = (D-1, (P,1) + shp)
        """
        D,P = self.tc.shape[:2]
        shp = self.tc.shape[2:]
        return UTPM(self.tc[1:,...].reshape((D-1,1) + (P,) + shp))
        
    def JTtoF(self):
        """
        inverse operation of FtoJT
        x.tc.shape = (D,1, P,shp)
        y = x.JTtoF()
        y.tc.shape = (D+1, P, shp)
        """
        D = self.tc.shape[0]
        P = self.tc.shape[2]
        shp = self.tc.shape[3:]
        tmp = numpy.zeros((D+1,P) + shp)
        tmp[0:D,...] = self.tc.reshape((D,P) + shp)
        return UTPM(tmp)        

    def clone(self):
        return UTPM(self.tc.copy())

    def get_shape(self):
        return numpy.shape(self.tc[0,0,...])
    shape = property(get_shape)
    
    def get_ndim(self):
        return numpy.ndim(self.tc[0,0,...])
    ndim = property(get_ndim)
    
    def reshape(self, dims):
        return UTPM(self.tc.reshape(self.tc.shape[0:2] + dims))

    def get_transpose(self):
        return self.transpose()
    def set_transpose(self,x):
        raise NotImplementedError('???')
    T = property(get_transpose, set_transpose)

    def transpose(self, axes = None):
        if axes != None:
            raise NotImplementedError('should implement that...')
        Nshp = len(self.shape)
        axes_ids = tuple(range(2,2+Nshp)[::-1])
        return UTPM( numpy.transpose(self.tc,axes=(0,1) + axes_ids))

    def set_zero(self):
        self.tc[...] = 0.
        return self

    def zeros_like(self):
        return UTPM(numpy.zeros_like(self.tc))
        

    def __str__(self):
        return str(self.tc)

    def __repr__(self):
        return self.__str__()
        
        
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
                
            out = cls(cls.__zeros__(out_shp))
            cls._dot( x.data, y.data, out = out.data)
            
        elif isinstance(x, UTPM) and not isinstance(y, UTPM):
            x_shp = x.data.shape
            y_shp = y.shape
            
            if  len(y_shp) == 1:
                out_shp = x_shp[:-1]
                
            else:
                out_shp = x_shp[:2] + x_shp[2:-1] + y_shp[:-2] + y_shp[-1:]
                
            out = cls(cls.__zeros__(out_shp))
            cls._dot_non_UTPM_y(x.data, y, out = out.data)
            
        elif not isinstance(x, UTPM) and isinstance(y, UTPM):
            x_shp = x.shape
            y_shp = y.data.shape
            
            if  len(y_shp[2:]) == 1:
                out_shp = y_shp[:2] + x_shp[:-1]
                
            else:
                out_shp = y_shp[:2] + x_shp[:-1] + y_shp[2:][:-2] + y_shp[2:][-1:]

            out = cls(cls.__zeros__(out_shp))
            cls._dot_non_UTPM_x(x, y.data, out = out.data)
            
            
        else:
            raise NotImplementedError('should implement that')
            
        return out
    
    @classmethod
    def inv(cls, A, out = None):
        if out == None:
            out = cls(cls.__zeros__(A.data.shape))
        else:
            raise NotImplementedError('')
        (D,P,N,M) = out.data.shape

        # tc[0] element
        for p in range(P):
            out.data[0,p,:,:] = numpy.linalg.inv(A.data[0,p,:,:])

        # tc[d] elements
        for d in range(1,D):
            for p in range(P):
                for c in range(1,d+1):
                    out.data[d,p,:,:] += numpy.dot(A.data[c,p,:,:], out.data[d-c,p,:,:],)
                out.data[d,p,:,:] =  numpy.dot(-out.data[0,p,:,:], out.data[d,p,:,:],)
        return out

    @classmethod
    def solve(cls, A, x, out = None):
        """
        solves for y in: A y = x
        
        """
        if isinstance(A, UTPM) and isinstance(x, UTPM):
            A_shp = numpy.shape(A.data)
            x_shp = numpy.shape(x.data)
    
            assert A_shp[:2] == x_shp[:2]
            if A_shp[2] != x_shp[2]:
                print ValueError('A.data.shape = %s does not match x.data.shape = %s'%(str(A_shp), str(x_shp)))
    
            D, P, M = A_shp[:3]
            
            if out == None:
                out = cls(cls.__zeros__((D,P,M) + x_shp[3:]))
    
            UTPM._solve(A.data, x.data, out = out.data)
        
        elif not isinstance(A, UTPM) and isinstance(x, UTPM):
            A_shp = numpy.shape(A)
            x_shp = numpy.shape(x.data)
            M = A_shp[0]
            D,P = x_shp[:2]
            out = cls(cls.__zeros__((D,P,M) + x_shp[3:]))
            cls._solve_non_UTPM_A(A, x.data, out = out.data)
            
        elif isinstance(A, UTPM) and not isinstance(x, UTPM):
            A_shp = numpy.shape(A.data)
            x_shp = numpy.shape(x)
            D,P,M = A_shp[:3]
            out = cls(cls.__zeros__((D,P,M) + x_shp[1:]))
            cls._solve_non_UTPM_x(A.data, x, out = out.data)
            
        else:
            raise NotImplementedError('should implement that')
            
        return out


    @classmethod
    def qr(cls, A, out = None):
        D,P,M,N = numpy.shape(A.data)
        K = min(M,N)
        
        if out == None:
            Q = cls(cls.__zeros__((D,P,M,K)))
            R = cls(cls.__zeros__((D,P,K,N)))

        UTPM._qr(Q.data, R.data, out = A.data)

        return Q,R
        
    @classmethod
    def eig(cls, A, out = None):
        """
        computes the eigenvalue decomposition A = Q^T L Q
        
        (l,Q) = UTPM.eig(A, out=None)
        
        """
        
        D,P,M,N = numpy.shape(A.data)
        
        if out == None:
            Q = cls(cls.__zeros__((D,P,N,N)))
            L = cls(cls.__zeros__((D,P,N)))
        
        UTPM._eig(Q.data, L.data, A.data)
      
        return L,Q
    
    @classmethod
    def _max(cls, x_data, axis = None, out = None):

        if out == None:
            raise NotImplementedError('should implement that')
        
        x_shp = x_data.shape

        D,P = x_shp[:2]
        shp = x_shp[2:]

        if len(shp) > 1:
            raise NotImplementedError('should implement that')
        
        for p in range(P):
            out[:,p] = x_data[:,p,numpy.argmax(x_data[0,p])]
            
    
    @classmethod
    def _idiv(cls, z_data, x_data):
        (D,P) = z_data.shape[:2]
        tmp_data = z_data.copy()
        for d in range(D):
            tmp_data[d,:,...] = 1./ x_data[0,:,...] * ( z_data[d,:,...] - numpy.sum(tmp_data[:d,:,...] * x_data[d:0:-1,:,...], axis=0))
        z_data[...] = tmp_data[...]
        
    @classmethod
    def _div(cls, x_data, y_data, out = None):
        """
        z = x/y
        """
        z_data = out
        if out == None:
            return NotImplementedError('')
        
        (D,P) = z_data.shape[:2]
        for d in range(D):
            z_data[d,:,...] = 1./ y_data[0,:,...] * ( x_data[d,:,...] - numpy.sum(z_data[:d,:,...] * y_data[d:0:-1,:,...], axis=0))
            
    @classmethod
    def _dot(cls, x_data, y_data, out = None):
        """
        z = dot(x,y)
        """
        
        if out == None:
            raise NotImplementedError('should implement that')
            
        z_data = out
        z_data[...] = 0.
            
        D,P = x_data.shape[:2]
        
        # print 'x_data.shape=', x_data.shape
        # print 'y_data.shape=', y_data.shape
        # print 'z_data.shape=', z_data.shape

        for d in range(D):
            for p in range(P):
                for c in range(d+1):
                    z_data[d,p,...] += numpy.dot(x_data[c,p,...], y_data[d-c,p,...])
                    
                    
        return out

    @classmethod
    def _dot_non_UTPM_y(cls, x_data, y_data, out = None):
        """
        z = dot(x,y)
        """
        
        if out == None:
            raise NotImplementedError('should implement that')
            
        z_data = out
        z_data[...] = 0.
        
        D,P = x_data.shape[:2]

        for d in range(D):
            for p in range(P):
                z_data[d,p,...] = numpy.dot(x_data[d,p,...], y_data[...])
                
        return out
                
    @classmethod
    def _dot_non_UTPM_x(cls, x_data, y_data, out = None):
        """
        z = dot(x,y)
        """
        
        if out == None:
            raise NotImplementedError('should implement that')
            
        z_data = out
        z_data[...] = 0.
        
        D,P = y_data.shape[:2]

        for d in range(D):
            for p in range(P):
                z_data[d,p,...] = numpy.dot(x_data[...], y_data[d,p,...])

        return out

    @classmethod
    def _solve(cls, A_data, x_data, out = None):
        """
        solves the linear system of equations for y::
            
            A y = x
        
        """
        
        if out == None:
            raise NotImplementedError('should implement that')
            
        y_data = out
        
        x_shp = numpy.shape(x_data)
        A_shp = numpy.shape(A_data)
        D,P,M,N = A_shp
        D,P,M,K = x_shp

        # d = 0:  base point
        for p in range(P):
            y_data[0,p,...] = numpy.linalg.solve(A_data[0,p,...], x_data[0,p,...])

        # d = 1,...,D-1
        tmp = numpy.zeros((M,K),dtype=float)
        for d in range(1, D):
            for p in range(P):
                tmp[:,:] = x_data[d,p,:,:]
                for k in range(1,d+1):
                    tmp[:,:] -= numpy.dot(A_data[k,p,:,:],y_data[d-k,p,:,:])
                y_data[d,p,:,:] = numpy.linalg.solve(A_data[0,p,:,:],tmp)
                
        return out
                
                
    @classmethod
    def _solve_non_UTPM_A(cls, A_data, x_data, out = None):
        """
        solves the linear system of equations for y::
            
            A y = x
        
        when A is a simple (N,N) float array
        """
        
        if out == None:
            raise NotImplementedError('should implement that')
            
        y_data = out
        
        x_shp = numpy.shape(x_data)
        A_shp = numpy.shape(A_data)
        M,N = A_shp
        D,P,M,K = x_shp
        
        assert M == N

        # d = 0:  base point
        for p in range(P):
            y_data[0,p,:,:] = numpy.linalg.solve(A_data[:,:], x_data[0,p,:,:])

        # d = 1,...,D-1
        tmp = numpy.zeros((M,K),dtype=float)
        for d in range(1, D):
            for p in range(P):
                tmp[:,:] = x_data[d,p,:,:]
                for k in range(1,d+1):
                    tmp[:,:] -= numpy.dot(A_data[...],y_data[d-k,p,:,:])
                y_data[d,p,:,:] = numpy.linalg.solve(A_data[:,:],tmp)
                
        return out

    @classmethod
    def _solve_non_UTPM_x(cls, A_data, x_data, out = None):
        """
        solves the linear system of equations for y::

            A y = x

        where x is simple (N,K) float array
        """
        
        if out == None:
            raise NotImplementedError('should implement that')
            
        y_data = out

        x_shp = numpy.shape(x_data)
        A_shp = numpy.shape(A_data)
        D,P,M,N = A_shp
        M,K = x_shp

        assert M==N

        # d = 0:  base point
        for p in range(P):
            y_data[0,p,...] = numpy.linalg.solve(A_data[0,p,...], x_data[...])

        # d = 1,...,D-1
        tmp = numpy.zeros((M,K),dtype=float)
        for d in range(1, D):
            for p in range(P):
                tmp[:,:] = 0.
                for k in range(1,d+1):
                    tmp[:,:] -= numpy.dot(A_data[k,p,:,:],y_data[d-k,p,:,:])
                y_data[d,p,:,:] = numpy.linalg.solve(A_data[0,p,:,:],tmp)
                
                
        return out
    

    @classmethod
    def __zeros_like__(cls, data):
        return numpy.zeros_like(data)
        
    @classmethod
    def __zeros__(cls, shp):
        return numpy.zeros(shp)

    @classmethod
    def _qr(cls, Q_data, R_data, out = None):
        """
        computes the qr decomposition (Q,R) = qr(A)    <===>    QR = A
        
        INPUTS:
            A_data      (D,P,M,N) array             regular matrix
            
        OUTPUTS:
            Q_data      (D,P,M,K) array             orthogonal vectors Q_1,...,Q_K
            R_data      (D,P,K,N) array             upper triagonal matrix
            
            where K = min(M,N)
        
        """
        
        if out == None:
            raise NotImplementedError('need to implement that...')
        A_data = out
        
        # input checks
        DT,P,M,N = numpy.shape(A_data)
        K = min(M,N)
        
        if Q_data.shape != (DT,P,M,K):
            raise ValueError('expected Q_data.shape = %s but provided %s'%(str((DT,P,M,K)),str(Q_data.shape)))
        assert R_data.shape == (DT,P,K,N)
        
        if not M >= N:
            raise NotImplementedError('A_data.shape = (DT,P,M,N) = %s but require (for now) that M>=N')
        
        # INIT: compute the base point
        for p in range(P):
            Q_data[0,p,:,:], R_data[0,p,:,:] = numpy.linalg.qr(A_data[0,p,:,:])
        
        dF = numpy.zeros((P,M,N))
        dG = numpy.zeros((P,K,K))
        X  = numpy.zeros((P,K,K))

        PL = numpy.array([[ r > c for c in range(N)] for r in range(K)],dtype=float)
        
        Rinv = numpy.zeros((P,K,N))
        for p in range(P):
            Rinv[p] = numpy.linalg.inv(R_data[0,p])
        
        # ITERATE: compute the derivatives
        for D in range(1,DT):
            # STEP 1:
            dF[...] = 0.
            dG[...] = 0
            X[...]  = 0

            for d in range(1,D):
                for p in range(P):
                    dF[p] += numpy.dot(Q_data[d,p,:,:], R_data[D-d,p,:,:])
                    dG[p] -= numpy.dot(Q_data[d,p,:,:].T, Q_data[D-d,p,:,:])
                    
            # STEP 2:
            H = A_data[D,:,:,:] - dF[:,:,:]
            S = - 0.5 * dG
            
            # STEP 3:
            for p in range(P):
                X[p,:,:] = PL * (numpy.dot( numpy.dot(Q_data[0,p,:,:].T, H[p,:,:,]), numpy.linalg.inv(R_data[0,p,:,:])) - S[p,:,:])
                X[p,:,:] = X[p,:,:] - X[p,:,:].T
                
            # STEP 4:
            K = S + X
            
            # STEP 5:
            for p in range(P):
                R_data[D,p,:,:] = numpy.dot(Q_data[0,p,:,:].T, H[p,:,:]) - numpy.dot(K[p,:,:],R_data[0,p,:,:])
                R_data[D,p,:,:] = R_data[D,p,:,:] - PL * R_data[D,p,:,:]
                
            # STEP 6:
            for p in range(P):
                Q_data[D,p,:,:] = numpy.dot(H[p] - numpy.dot(Q_data[0,p],R_data[D,p]), Rinv[p]) #numpy.dot(Q_data[0,p,:,:],K[p,:,:])
        
    @classmethod
    def _eig(cls, Q_data, L_data, A_data):
        """
        computes the eigenvalue decompositon

        L,Q = eig(A)

        for symmetric matrix A with distinct eigenvalues, i.e. 
        where L is a diagonal matrix of ordered eigenvalues l_1 > l_2 > ...> l_N
        and Q a matrix of corresponding orthogonal eigenvectors

        """
        # input checks
        DT,P,M,N = numpy.shape(A_data)
        
        assert M == N

        if Q_data.shape != (DT,P,N,N):
            raise ValueError('expected Q_data.shape = %s but provided %s'%(str((DT,P,M,K)),str(Q_data.shape)))

        if L_data.shape != (DT,P,N):
            raise ValueError('expected L_data.shape = %s but provided %s'%(str((DT,P,N)),str(L_data.shape)))


        # INIT: compute the base point
        for p in range(P):
            L_data[0,p,:], Q_data[0,p,:,:] = numpy.linalg.eig(A_data[0,p,:,:])

        Id = numpy.zeros((P,N,N))
        for p in range(P):
            Id[p] = numpy.eye(N)

        # save zero'th coefficient of L_data as diagonal matrix
        L = numpy.zeros((P,N,N))
        for p in range(P):
            L[p] = numpy.diag(L_data[0,p])
            
        dG = numpy.zeros((P,N,N))

        # ITERATE: compute derivatives
        for D in range(1,DT):
            # print 'D=',D
            dG[...] = 0.
            
            dL = numpy.zeros((P,N,N))

            # STEP 1:
            dF = truncated_triple_dot(Q_data.transpose(0,1,3,2), A_data, Q_data, D)
                
            for d in range(1,D):
                dG += vdot(Q_data[d,...].transpose(0,2,1), Q_data[D-d,...])

            # STEP 2:
            S = -0.5 * dG

            # STEP 3:
            K = dF + vdot(vdot(Q_data.transpose(0,1,3,2)[0], A_data[D]),Q_data[0]) + \
                vdot(S, L) + vdot(L,S)
                
            # STEP 4:
            dL = Id * K
                
            # STEP 5:
            H = numpy.zeros((P,N,N),dtype=float)
            for p in range(P):
                for r in range(N):
                    for c in range(N):
                        if c == r:
                            continue
                        H[p,r,c] = 1./( L[p,c,c] - L[p,r,r])
                        
            # STEP 6:
            tmp0 = K - dL
            tmp1 = H * tmp0
            tmp2 = tmp1 + S
            Q_data[D] = vdot(Q_data[0], tmp2)
            
            # STEP 7:
            for p in range(P):
                L_data[D,p,:] = numpy.diag(dL[p])
    
    @classmethod
    def _mul_non_UTPM_x(cls, x_data, y_data, out = None):
        """
        z = x * y
        """
        
        if out == None:
            raise NotImplementedError('need to implement that...')
        z_data = out
        
        D,P = numpy.shape(y_data)[:2]
        
        for d in range(D):
            for p in range(P):
                z_data[d,p] = x_data * y_data[d,p]
                
    @classmethod
    def _eig_pullback(cls, Qbar_data, lambar_data, A_data, Q_data, lam_data, out = None):
        
        if out == None:
            raise NotImplementedError('need to implement that...')
            
        Abar_data = out

        A_shp = A_data.shape
        D,P,M,N = A_shp
        
        assert M == N
        
        # allocating temporary storage
        H = numpy.zeros(A_shp)
        tmp1 = numpy.zeros((D,P,N,N), dtype=float)
        tmp2 = numpy.zeros((D,P,N,N), dtype=float)
        
        Id = numpy.zeros((D,P,1))
        Id[0,:,0] = numpy.eye(P)
        
        Lam_data    = cls._diag(lam_data)
        Lambar_data = cls._diag(lambar_data)
        
        # STEP 1: compute H
        for m in range(N):
            for n in range(N):
                if n == m:
                    continue
                tmp = lam_data[:,:,n] -   lam_data[:,:,m]
                cls._div(Id, tmp, out = H[:,:,m,n])
        
        # STEP 2: compute Lbar +  H * Q^T Qbar
        cls._dot(cls._transpose(Q_data), Qbar_data, out = tmp1)
        tmp1[...] *= H[...]
        tmp1[...] += Lambar_data[...]
        
        # STEP 3: compute Q ( Lbar +  H * Q^T Qbar ) Q^T
        cls._dot(Q_data, tmp1, out = tmp2)
        cls._dot(tmp2, cls._transpose(Q_data), out = tmp1)
        
        Abar_data += tmp1
        
        return out
        
    
    @classmethod
    def qr_pullback(cls, Qbar, Rbar, A, Q, R):
        
        A_shp = A.shape
        M,N = A_shp
        
        PL  = numpy.array([[ c < r for c in range(N)] for r in range(N)],dtype=float)
        
        V = cls.dot(Qbar.T, Q) - cls.dot(R, Rbar.T)
        tmp = (V.T - V) * PL
        RTinv = UTPM.inv(R.T)
        tmp = cls.dot(tmp, RTinv)
        return cls.dot(Q, Rbar + tmp) + cls.dot((Qbar - cls.dot(Q, cls.dot(Q.T,Qbar))),RTinv)
        

    @classmethod
    def _qr_pullback(cls,Qbar_data, Rbar_data, A_data, Q_data, R_data, out = None):
        
        if out == None:
            raise NotImplementedError('need to implement that...')
            
        Abar_data = out
        
        A_shp = A_data.shape
        D,P,M,N = A_shp
        
        
        if M < N:
            raise ValueError('supplied matrix has more columns that rows')

        # allocate temporary storage and temporary matrices
        tmp1 = numpy.zeros((D,P,N,N))
        tmp2 = numpy.zeros((D,P,N,N))
        tmp3 = numpy.zeros((D,P,M,N))
        tmp4 = numpy.zeros((D,P,M,N))
        PL  = numpy.array([[ c < r for c in range(N)] for r in range(N)],dtype=float)
        
        # STEP 1: compute V
        cls._dot( cls._transpose(Qbar_data), Q_data, out = tmp1)
        cls._dot( R_data, cls._transpose(Rbar_data), out = tmp2)
        tmp1[...] -= tmp2[...]
        
        # STEP 2: compute PL * (V.T - V)
        tmp2[...]  = cls._transpose(tmp1)
        tmp2[...] -= tmp1[...]
        cls._mul_non_UTPM_x(PL, tmp2, out = tmp1)
        
        # STEP 3: compute PL * (V.T - V) R^{-T}
        cls._solve(R_data, cls._transpose(tmp1), out = tmp2)
        tmp2 = tmp2.transpose((0,1,3,2))
        
        # STEP 4: compute Rbar + PL * (V.T - V) R^{-T}
        tmp2[...] += Rbar_data[...]
        
        # STEP 5: compute Q ( Rbar + PL * (V.T - V) R^{-T} )
        cls._dot( Q_data, tmp2, out = tmp3)
        Abar_data += tmp3
        
        if M > N:
            # STEP 6: compute (Qbar - Q Q^T Qbar) R^{-T}
            cls._dot( cls._transpose(Q_data), Qbar_data, out = tmp1)
            cls._dot( Q_data, tmp1, out = tmp3)
            tmp3 *= -1.
            tmp3 += Qbar_data
            cls._solve(R_data, cls._transpose(tmp3), out = cls._transpose(tmp4))
            Abar_data += tmp4
        
        return out
        
    @classmethod
    def _transpose(cls, a_data, axes = None):
        """Permute the dimensions of UTPM data"""
        if axes != None:
            raise NotImplementedError('should implement that')
        
        return numpy.transpose(a_data, (0,1,3,2))
        
    @classmethod
    def _diag(cls, v_data, k = 0, out = None):
        """Extract a diagonal or construct  diagonal UTPM data"""
        
        if numpy.ndim(v_data) == 3:
            D,P,N = v_data.shape
            if out == None:
                out = numpy.zeros((D,P,N,N),dtype=float)
            else:
                out[...] = 0.
                
            for d in range(D):
                for p in range(P):
                    out[d,p] = numpy.diag(v_data[d,p])
                        
            return out
            
        else:
            raise NotImplementedError('should implement that')
        



