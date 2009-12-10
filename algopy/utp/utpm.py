"""
Implementation of the univariate matrix polynomial.
The algebraic class is

M[t]/<t^D>

where M is the ring of matrices and t in R.

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
        return x.inv()
    else:
        return numpy.linalg.inv(x)
        
def dot(x,y):
    if isinstance(x, UTPM):
        if not isinstance(y, UTPM):
            raise NotImplementedError('dot currently only implemented for x,y UTPM instances')
        return x.dot(y)
    else:
        return numpy.dot(x,y)

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
    
    def __init__(self, X, Xdot = None, shift = 0):
        """ INPUT:	shape([X]) = (D,P,N,M)
        
        shift>0 is necessary for the reverse mode.
        Example:
        In the forward mode compute
            y(t) = d/dt x(t)
        In the reverse mode compute
        
            ybar(t) d y(t) = ybar(t) d/dt d y(t)
            
        I.e. that means that ybar_shift(t) = ybar(t) d/dt is another adjoint operator.
        Multiplying with this operator, i.e. 
            ybar(t) = [yb0,yb1]
            y(t)    = [y0,y1,y2]
            
            ybar_shift(t) * d y(t) = [yb0 * dy1, yb0 * dy2 + yb1 * dy1]
        
        """
        Ndim = numpy.ndim(X)
        self.shift = shift
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
                    self.tc[d,p,...] *= rhs.tc[self.shift,p,...]
                    for c in range(d):
                        self.tc[d,p,...] += self.tc[c,p,...] * rhs.tc[d-c + self.shift,p,...]
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

    def dot(self,rhs):
        D,P = self.tc.shape[:2]
        
        if len(self.shape) == 1 and len(rhs.shape) == 1:
            tc = numpy.zeros((D,P,1))
            
        elif len(self.shape) == 2 and len(rhs.shape) == 1 :
            tc = numpy.zeros((D,P,self.shape[0]))
            
        elif  len(self.shape) == 1 and len(rhs.shape) == 2:
            tc = numpy.zeros((D,P,rhs.shape[1]))

        elif  len(self.shape) == 2 and len(rhs.shape) == 2:
            tc = numpy.zeros((D,P, self.shape[0],rhs.shape[1]))
            
        elif self.ndim == 0 and rhs.ndim == 0:
            tc = numpy.zeros((D,P))
            
        else:
            raise NotImplementedError('you tried to dot(%s,%s)'%(str(self.shape),str(rhs.shape))) 
            
        retval = UTPM(tc)
        for d in range(D):
            for p in range(P):
                for c in range(d+1):
                    retval.tc[d,p,...] += numpy.dot(self.tc[c,p,...], rhs.tc[d-c,p,...])
        return retval

    def inv(self):
        retval = UTPM(numpy.zeros(numpy.shape(self.tc)))
        (D,P,N,M) = numpy.shape(retval.tc)

        # tc[0] element
        for p in range(P):
            retval.tc[0,p,:,:] = numpy.linalg.inv(self.tc[0,p,:,:])

        # tc[d] elements
        for d in range(1,D):
            for p in range(P):
                for c in range(1,d+1):
                    retval.tc[d,p,:,:] += numpy.dot(self.tc[c,p,:,:], retval.tc[d-c,p,:,:],)
                retval.tc[d,p,:,:] =  numpy.dot(-retval.tc[0,p,:,:], retval.tc[d,p,:,:],)
        return retval

    def solve(self,A):
        """
        A y = x  <=> y = solve(A,x)
        is implemented here as y = x.solve(A)
        """
        retval = UTPM( numpy.zeros( numpy.shape(self.tc)))
        (D,P,N,M) = numpy.shape(retval.tc)
        assert M == 1
        tmp = numpy.zeros((N,M),dtype=float)
        for d in range(D):
            for p in range(P):
                tmp[:,:] = self.tc[d,p,:,:]
                for k in range(1,d+1):
                    tmp[:,:] -= numpy.dot(A.tc[k,p,:,:],retval.tc[d-k,p,:,:])
                retval.tc[d,p,:,:] = numpy.linalg.solve(A.tc[0,p,:,:],tmp)
        return retval

    @classmethod
    def __zeros_like__(cls, data):
        return numpy.zeros_like(data)
        
    @classmethod
    def __zeros__(cls, shp):
        return numpy.zeros(shp)

    def qr(self):
        Q = self.__class__(self.__class__.__zeros_like__(self.data))
        R = self.__class__(self.__class__.__zeros_like__(self.data))

        UTPM.cls_qr(Q.data, R.data, self.data)

        return Q,R
    
    @classmethod
    def cls_qr(cls, Q_data, R_data, A_data):
        """
        computes the qr decomposition (Q,R) = qr(A)    <===>    QR = A
        
        INPUTS:
            A_data      (D,P,N,N) array             regular matrix
            
        OUTPUTS:
            Q_data      (D,P,N,N) array             orthogonal matrix Q_1,...,Q_N
            R_data      (D,P,N,N) array             upper triagonal matrix
        
        """
        DT,P,N,N = numpy.shape(A_data)
        
        # QR decomposition at the base point, i.e. for order D=1
        
        for p in range(P):
            Q_data[0,p,:,:], R_data[0,p,:,:] = numpy.linalg.qr(A_data[0,p,:,:])
        
        dF = numpy.zeros((P,N,N))
        dG = numpy.zeros((P,N,N))
        X  = numpy.zeros((P,N,N))

        PL = numpy.array([[ r > c for c in range(N)] for r in range(N)],dtype=float)
        
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
            S = - 0.5 * dF
            
            # STEP 3:
            for p in range(P):
                X[p,:,:] = PL * (numpy.dot( numpy.dot(Q_data[0,p,:,:].T, H[p,:,:,]), numpy.linalg.inv(R_data[0,p,:,:])) - S[p,:,:])
                X[p,:,:] = X[p,:,:] - X[p,:,:].T
                
            # STEP 4:
            K = S + X
            
            # STEP 5:
            for p in range(P):
                Q_data[D,p,:,:] = numpy.dot(Q_data[0,p,:,:],K[p,:,:])
                R_data[D,p,:,:] = numpy.dot(Q_data[0,p,:,:].T, H[p,:,:]) - numpy.dot(K[p,:,:],R_data[0,p,:,:])
                
                R_data[D,p,:,:] = R_data[D,p,:,:] - PL * R_data[D,p,:,:]
   

    def qr_rectangular(self):
        D,P,M,N = numpy.shape(self.data)
        K = min(M,N)
        
        Q = self.__class__(self.__class__.__zeros__((D,P,M,K)))
        R = self.__class__(self.__class__.__zeros__((D,P,K,N)))

        UTPM.cls_qr_rectangular(Q.data, R.data, self.data)

        return Q,R

   
    @classmethod
    def cls_qr_rectangular(cls, Q_data, R_data, A_data):
        """
        computes the qr decomposition (Q,R) = qr(A)    <===>    QR = A
        
        INPUTS:
            A_data      (D,P,M,N) array             regular matrix
            
        OUTPUTS:
            Q_data      (D,P,M,K) array             orthogonal vectors Q_1,...,Q_K
            R_data      (D,P,K,N) array             upper triagonal matrix
            
            where K = min(M,N)
        
        """
        
        # input checks
        DT,P,M,N = numpy.shape(A_data)
        K = min(M,N)
        
        if Q_data.shape != (DT,P,M,K):
            raise ValueError('expected Q_data.shape = %s but provided %s'%(str((DT,P,M,K)),str(Q_data.shape)))
        assert R_data.shape == (DT,P,K,N)
        
        assert M >= N
        
        # INIT: compute the base point
        for p in range(P):
            Q_data[0,p,:,:], R_data[0,p,:,:] = numpy.linalg.qr(A_data[0,p,:,:])
        
        # dF = numpy.zeros((P,N,N))
        # dG = numpy.zeros((P,N,N))
        # X  = numpy.zeros((P,N,N))

        # PL = numpy.array([[ r > c for c in range(N)] for r in range(N)],dtype=float)
        
        # # ITERATE: compute the derivatives
        # for D in range(1,DT):
            # # STEP 1:
            # dF[...] = 0.
            # dG[...] = 0
            # X[...]  = 0

            # for d in range(1,D):
                # for p in range(P):
                    # dF[p] += numpy.dot(Q_data[d,p,:,:], R_data[D-d,p,:,:])
                    # dG[p] -= numpy.dot(Q_data[d,p,:,:].T, Q_data[D-d,p,:,:])
                    
            # # STEP 2:
            # H = A_data[D,:,:,:] - dF[:,:,:]
            # S = - 0.5 * dF
            
            # # STEP 3:
            # for p in range(P):
                # X[p,:,:] = PL * (numpy.dot( numpy.dot(Q_data[0,p,:,:].T, H[p,:,:,]), numpy.linalg.inv(R_data[0,p,:,:])) - S[p,:,:])
                # X[p,:,:] = X[p,:,:] - X[p,:,:].T
                
            # # STEP 4:
            # K = S + X
            
            # # STEP 5:
            # for p in range(P):
                # Q_data[D,p,:,:] = numpy.dot(Q_data[0,p,:,:],K[p,:,:])
                # R_data[D,p,:,:] = numpy.dot(Q_data[0,p,:,:].T, H[p,:,:]) - numpy.dot(K[p,:,:],R_data[0,p,:,:])
                
                # R_data[D,p,:,:] = R_data[D,p,:,:] - PL * R_data[D,p,:,:]

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
        return UTPM(self.tc.copy(), shift = self.shift)

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

