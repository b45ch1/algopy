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
        
        print 'z_data=',z_data

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
    def _dot_pullback(cls, zbar_data, x_data, y_data, z_data, out = None):
        if out == None:
            raise NotImplementedError('should implement that')
        
        (xbar_data, ybar_data) = out
        
        xbar_data += cls._dot(zbar_data, cls._transpose(y_data), out = xbar_data.copy())
        ybar_data += cls._dot(cls._transpose(x_data), zbar_data, out = ybar_data.copy())
        return out
        

    @classmethod
    def _inv(cls, x_data, out = None):
        """
        computes y = inv(x)
        """
        
        if out == None:
            raise NotImplementedError('should implement that')
        
        y_data, = out
        (D,P,N,M) = y_data.shape
            
        # tc[0] element
        for p in range(P):
            y_data[0,p,:,:] = numpy.linalg.inv(x_data[0,p,:,:])

        # tc[d] elements
        for d in range(1,D):
            for p in range(P):
                for c in range(1,d+1):
                    y_data[d,p,:,:] += numpy.dot(x_data[c,p,:,:], y_data[d-c,p,:,:],)
                y_data[d,p,:,:] =  numpy.dot(-y_data[0,p,:,:], y_data[d,p,:,:],)
        return y_data
        

    @classmethod
    def _inv_pullback(cls, ybar_data, x_data, y_data, out = None):
        if out == None:
            raise NotImplementedError('should implement that')
        
        xbar_data = out
        tmp1 = numpy.zeros(xbar_data.shape)
        tmp2 = numpy.zeros(xbar_data.shape)
        
        tmp1 = cls._dot(ybar_data, cls._transpose(y_data), out = tmp1)
        tmp2 = cls._dot(cls._transpose(y_data), tmp1, out = tmp2)

        xbar_data -= tmp2
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

        for d in range(D):
            for p in range(P):
                y_data[d,p,...] = numpy.linalg.solve(A_data[:,:], x_data[d,p,...])

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
    def _choleksy(cls, A_data, L_data):
        """
        compute the choleksy decomposition in Taylor arithmetic of a symmetric
        positive definite matrix A
        i.e.
        ..math:
        
            A = L L^T
        """
        DT,P,N = numpy.shape(A_data)[:3]
        
        # allocate (temporary) projection matrix
        Proj = numpy.zeros((N,N))
        for r in range(N):
            for c in range(r+1):
                if r == c:
                    Proj[r,c] = 0.5
                else:
                    Proj[r,c] = 1
        
        for p in range(P):
            
            # base point: d = 0
            L_data[0,p] = numpy.linalg.cholesky(A_data[0,p])
            
            # allocate temporary storage
            L0inv = numpy.linalg.inv(L_data[0,p])
            dF    = numpy.zeros((N,N),dtype=float)
            
            # higher order coefficients: d > 0
            # STEP 1: compute diagonal elements of dL
            for D in range(1,DT):
                dF *= 0
                for d in range(1,D):
                    dF += numpy.dot(L_data[D-d,p], L_data[d,p].T)
                
                # print numpy.dot(L_data[1,p],L_data[1,p].T)
                # print 'dF = ',dF
                    
                dF -= A_data[D,p]
                
                dF = numpy.dot(numpy.dot(L0inv,dF),L0inv.T)
                
                # compute off-diagonal entries
                L_data[D,p] = - numpy.dot( L_data[0,p], Proj * dF)

                # compute diagonal entries
                tmp1 = numpy.diag(L_data[0,p])
                tmp2 = numpy.diag(dF)
                tmp3 = -0.5 * tmp1 * tmp2
                L_data[D,p][numpy.diag_indices(N)] = tmp3


    @classmethod
    def build_PL(cls, N):
        """
        build lower triangular matrix with all ones, i.e.
        
        PL = [[0,0,0],
              [1,0,0],
              [1,1,0]]
        """
        retval = numpy.zeros((N,N))
        
        for r in range(N):
            for c in range(r):
                retval[r,c] = 1.

        return retval

    @classmethod
    def build_PU(cls, N):
        """
        build upper triangular matrix with all ones, i.e.
        
        PL = [[0,1,1],
              [0,0,1],
              [0,0,0]]
        """
        retval = numpy.zeros((N,N))
        
        for r in range(N):
            for c in range(r+1,N):
                retval[r,c] = 1.
                
        return retval
        

    @classmethod
    def _pb_cholesky(cls, Lbar_data, A_data, L_data, out = None):
        """
        pullback of the linear form of the cholesky decomposition
        """
    
        if out == None:
            raise NotImplementedError('should implement this')
        
        Abar_data = out
        
        D,P,N = A_data.shape[:3]
        
        # compute (P_L + 0.5*P_D) * dot(L.T, Lbar)
        Proj = cls.build_PL(N) + 0.5 * numpy.eye(N)
        
        # compute (P_L + 0.5*P_D) * dot(L.T, Lbar)
        Proj = cls.build_PL(N) + 0.5 * numpy.eye(N)
        tmp = cls._dot(cls._transpose(L_data), Lbar_data, cls.__zeros_like__(A_data))
        tmp *= Proj
        
        # symmetrize (P_L + 0.5*P_D) * dot(L.T, Lbar)
        tmp = 0.5*(cls._transpose(tmp) + tmp)
        
        # compute Abar
        Linv_data = cls._inv(L_data, (cls.__zeros_like__(A_data),))
        tmp2 = cls._dot(cls._transpose(Linv_data), tmp, cls.__zeros_like__(A_data))
        tmp3 = cls._dot(tmp2, Linv_data, cls.__zeros_like__(A_data))
        Abar_data += tmp3
        
        return Abar_data
        

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
        
        # check if the output array is provided
        if out == None:
            raise NotImplementedError('need to implement that...')
        Q_data = out[0]
        R_data = out[1]
        
        DT,P,M,N = numpy.shape(A_data)
        K = min(M,N)
        
        if M < N:
            A1_data = A_data[:,:,:,:M]
            A2_data = A_data[:,:,:,M:]
            R1_data = R_data[:,:,:,:M]
            R2_data = R_data[:,:,:,M:]
            
            cls._qr_rectangular(A1_data, out = (Q_data, R1_data))
            cls._dot(Q_data.transpose((0,1,3,2)), A2_data, out=R2_data)
            
        else:
            cls._qr_rectangular(A_data, out = (Q_data, R_data))

    @classmethod
    def _qr_rectangular(cls,  A_data, out = None,  work = None):
        """
        computation of qr(A) where A.shape(M,N) with M >= N
        
        this function is called by the more general function _qr
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
            S =  0.5 * dG

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
    def _eigh(cls, L_data, Q_data, A_data, epsilon = 10**-8, full_output = False):
        """
        computes the eigenvalue decompositon

        L,Q = eig(A)

        for symmetric matrix A with possibly repeated eigenvalues, i.e.
        where L is a diagonal matrix of ordered eigenvalues l_1 >= l_2 >= ...>= l_N
        and Q a matrix of corresponding orthogonal eigenvectors

        """
        
        def find_repeated_values(L):
            """
            INPUT:  L    (N,) array of ordered values, dtype = float
            OUTPUT: b    (Nb,) array s.t. L[b[i:i+1]] are all repeated values
            
            Nb is the number of blocks of repeated values. It holds that
            b[-1] = N.
            
            e.g. L = [1.,1.,1.,2.,2.,3.,5.,7.,7.]
            then the output is [0,3,5,6,7,9]
            """
            N = len(L)
            # print 'L=',L
            b = [0]
            n = 0
            while n < N:
                m = n + 1
                while m < N:
                    # print 'n,m=',n,m
                    tmp = L[n] - L[m]
                    if numpy.abs(tmp) > epsilon:
                        b += [m]
                        break
                    m += 1
                n += (m - n)
            b += [N]
            
            # print 'OK'
            return numpy.asarray(b)
                            
        def generate_mask(blocks1, blocks2, nb):
            """
            
            e.g. blocks1 = [0,3,7,9]
                 blocks2 = [0,2,3,7,9]
            
            and nb = 1
                 
            then the corresponding matrix looks like
            
            mask = [[0,0,1],
                    [0,0,1],
                    [1,1,0]]
            """
            
            start1 = blocks1[nb]
            stop1  = blocks1[nb+1]
            
            start2, = numpy.where( start1 == blocks2 )
            print 'start2=',start2
            
            
            
            
        
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

        # save zero'th coefficient of L_data as diagonal matrix
        L = numpy.zeros((P,N,N))
        for p in range(P):
            L[p] = numpy.diag(L_data[0,p])
            
        # store blocks of repeated eigenvalues for all degrees in the variable blocks
        blocks_list = []
        tmp = []
        for p in range(P):
            tmp.append(find_repeated_values(L_data[0,p]))
        blocks_list.append(tmp)

        # compute H
        H = numpy.zeros((P,N,N))
        for p in range(P):
            for r in range(N):
                for c in range(N):
                    tmp = L_data[0,p,c] - L_data[0,p,r]
                    if abs(tmp) > epsilon:
                        H[p,r,c] = 1./tmp

        dG = numpy.zeros((P,N,N))

        # ITERATE: compute derivatives
        for D in range(1,DT):
            print 'D=',D
            dG[...] = 0.

            # STEP 1:
            dF = truncated_triple_dot(Q_data.transpose(0,1,3,2), A_data, Q_data, D)

            for d in range(1,D):
                dG += vdot(Q_data[d,...].transpose(0,2,1), Q_data[D-d,...])

            # STEP 2:
            S = -0.5 * dG

            # STEP 3:
            K = dF + vdot(vdot(Q_data.transpose(0,1,3,2)[0], A_data[D]),Q_data[0]) + \
                vdot(S, L) + vdot(L,S)

            # STEP 4: compute Q
            XT = K*H
            for p in range(P):
                Q_data[D,p] = numpy.dot(Q_data[0,p], XT[p] + S[p])
            
            # STEP 5: eigenvalue decomposition of dL in the invariant subspace
            for p in range(P):
                blocks = blocks_list[D-1][p]
                for nb in range(len(blocks)-1):
                    start, stop = blocks[nb], blocks[nb+1]
                    
                    # print generate_mask(blocks_list[0][p], blocks, 0)
                    L_data[D,p,start:stop], U = numpy.linalg.eigh(K[p,start:stop,start:stop])
                    
                    # print 'start,stop=',start,stop
                    # print 'U=', numpy.dot(U.T,U)
                    for d in range(D+1):
                        Q_data[d,p,:,start:stop] = numpy.dot(Q_data[d,p,:,start:stop], U)
            
            
            # STEP 6: update the blocks_list
            tmp = []
            for p in range(P):
                blocks = blocks_list[D-1][p]
                tmp2 = []
                for nb in range(len(blocks)-1):
                    start, stop = blocks[nb], blocks[nb+1]
                    tmp2.append(find_repeated_values(L_data[D,p,start:stop]) + start)
                tmp.append( numpy.unique(numpy.concatenate(tmp2)))
            blocks_list.append(tmp)
        
        print 'blocks_list=',blocks_list
        if full_output == True:
            return L_data, Q_data, blocks_list


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
                for p in range(P):
                    tmp = lam_data[0,p,n] - lam_data[0,p,m]
                    if numpy.abs(tmp) > 10**-8:
                        for d in range(D):
                            H[d,p,m,n] = 1./tmp
                # tmp = lam_data[:,:,n] -   lam_data[:,:,m]
                # cls._div(Id, tmp, out = H[:,:,m,n])

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
