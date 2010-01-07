import numpy



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



class RawAlgorithmsMixIn:
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
    def _argmax(cls, a_data, axis = None):

        if axis != None:
            raise NotImplementedError('should implement that')

        a_shp = a_data.shape
        D,P = a_shp[:2]
        return numpy.argmax(a_data[0].reshape((P,numpy.prod(a_shp[2:]))), axis = 1)


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
    def _solve_pullback(cls, ybar_data, A_data, x_data, y_data, out = None):

        if out == None:
            raise NotImplementedError('should implement that')

        Abar_data = out[0]
        xbar_data = out[1]

        tmp = numpy.zeros(xbar_data.shape)
        
        cls._solve( A_data.transpose((0,1,3,2)), ybar_data, out = tmp)

        xbar_data += tmp

        tmp *= -1.
        cls._iouter(tmp, y_data, Abar_data)

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

        x_shp = x_data.shape
        A_shp = A_data.shape
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
    def _ndim(cls, a_data):
        return a_data[0,0].ndim

    @classmethod
    def _shape(cls, a_data):
        return a_data[0,0].shape
        
    @classmethod
    def _reshape(cls, a_data, newshape, order = 'C'):

        if order != 'C':
            raise NotImplementedError('should implement that')

        return numpy.reshape(a_data, a_data.shape[:2] + newshape)

    @classmethod
    def _iouter(cls, x_data, y_data, out_data):
        """
        computes dyadic product and adds it to out
        out += x y^T
        """

        if len(cls._shape(x_data)) == 1:
            x_data = cls._reshape(x_data, cls._shape(x_data) + (1,))
        
        if len(cls._shape(y_data)) == 1:
            y_data = cls._reshape(y_data, cls._shape(y_data) + (1,))

        tmp = cls.__zeros__(out_data.shape)
        cls._dot(x_data, cls._transpose(y_data), out = tmp)

        out_data += tmp

        return out_data



    @classmethod
    def __zeros_like__(cls, data):
        return numpy.zeros_like(data)

    @classmethod
    def __zeros__(cls, shp):
        return numpy.zeros(shp)

    @classmethod
    def _qr(cls,  A_data, out = None,  work = None):
        """
        computes the qr decomposition (Q,R) = qr(A)    <===>    QR = A

        INPUTS:
            A_data      (D,P,M,N) array             regular matrix

        OUTPUTS:
            Q_data      (D,P,M,K) array             orthogonal vectors Q_1,...,Q_K
            R_data      (D,P,K,N) array             upper triagonal matrix

            where K = min(M,N)

        """
        
        DT,P,M,N = numpy.shape(A_data)
        K = min(M,N)        
        
        # check if the output array is provided
        if out == None:
            raise NotImplementedError('need to implement that...')
        Q_data = out[0]
        R_data = out[1]
        
        # input checks
        if Q_data.shape != (DT,P,M,K):
            raise ValueError('expected Q_data.shape = %s but provided %s'%(str((DT,P,M,K)),str(Q_data.shape)))
        assert R_data.shape == (DT,P,K,N)

        if not M >= N:
            raise NotImplementedError('A_data.shape = (DT,P,M,N) = %s but require (for now) that M>=N')        
                
        
        # check if work arrays are provided, if not allocate them
        if work == None:
            dF = numpy.zeros((P,M,N))
            dG = numpy.zeros((P,K,K))
            X  = numpy.zeros((P,K,K))
            PL = numpy.array([[ r > c for c in range(N)] for r in range(K)],dtype=float)
            Rinv = numpy.zeros((P,K,N))
            
        else:
            raise NotImplementedError('need to implement that...')


        # INIT: compute the base point
        for p in range(P):
            Q_data[0,p,:,:], R_data[0,p,:,:] = numpy.linalg.qr(A_data[0,p,:,:])


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
    def _eigh(cls, L_data, Q_data, A_data):
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
            L_data[0,p,:], Q_data[0,p,:,:] = numpy.linalg.eigh(A_data[0,p,:,:])

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
    def _eigh_pullback(cls, lambar_data, Qbar_data, A_data, lam_data, Q_data, out = None):

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

        Id = numpy.zeros((D,P))
        Id[0,:] = 1

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
    def _qr_pullback(cls, Qbar_data, Rbar_data, A_data, Q_data, R_data, out = None):

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

        Nshp = len(a_data.shape)
        axes_ids = tuple(range(2,Nshp)[::-1])
        return numpy.transpose(a_data,axes=(0,1) + axes_ids)

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
