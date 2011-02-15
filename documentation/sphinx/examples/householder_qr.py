import algopy, numpy

def house(x):
    """ computes the Householder vector v and twice its norm beta
    
    (v,beta) = house(x)
    
    Parameters
    ----------
    x: array_like
        len(x) = N
    
    Returns
    -------
    v: array_like
        len(v) = N
    
    beta: Float
        two times the 2-norm of v
        
    Description
    -----------
    computes beta and v to be used in the Householder reflector
    H(v) = 1 - beta dot(v,v.T)
    where v[0] = 1
    such that H(v)x = alpha * e_1
    i.e., H(v)x is a multiple of the first Cartesian basis vector
    """
    
    sigma = algopy.sqrt(algopy.dot(x.T,x))[0,0]
    
    v = x.copy()
    if x[0] <= 0:
        v[0] -= sigma

    else:
        v[0] += sigma
    
    v = v/v[0]
    beta = 2./algopy.dot(v.T,v)[0,0]
    
    return v, beta
    

def qr_house(A):
    """ computes QR decomposition using Householder relections
    
    (Q,R) = qr_house(A)
    
    such that 
    0 = Q R - A
    0 = dot(Q.T,Q) - eye(M)
    R upper triangular
    
    Parameters
    ----------
    A: array_like
       shape(A) = (M, N), M >= N
       overwritten on exit
    
    Returns
    -------
    R: array_like
        strict lower triangular part contains the Householder vectors v
        upper triangular matrix R
    
    Q: array_like
        orthogonal matrix

    """
    
    M,N = A.shape
    Q = algopy.zeros((M,M),dtype=A)
    Q += numpy.eye(M)
    H = algopy.zeros((M,M),dtype=A)
    for n in range(N):
        v,beta = house(A[n:,n:n+1])
        A[n:,n:] -= beta * algopy.dot(v, algopy.dot(v.T,A[n:,n:]))
        H[...] = numpy.eye(M)
        H[n:,n:] -= beta * algopy.dot(v,v.T)
        Q = algopy.dot(Q,H)
        
    return Q, algopy.triu(A)
    
# def build_Q(betas, A, S):
#     """ computes orthogonal matrix 
    
#     Parameters
#     ----------
#     betas: array_like
#         b
    
    
    
#     """
    
    
# def pb_qr_house(A, Abar, Q, Qbar, R, Rbar):
#     """ computes QR decomposition using Householder relections
    
#     (Q,R) = qr_house(A)
    
#     such that 
#     0 = Q R - A
#     0 = dot(Q.T,Q) - eye(M)
#     R upper triangular
    
#     Parameters
#     ----------
#     A: array_like
#        shape(A) = (M, N), M >= N
#        overwritten on exit
    
#     Returns:
#     R: array_like
#         strict lower triangular part contains the Householder vectors v
#         upper triangular matrix R
    
#     Q: array_like
#         orthogonal matrix

#     """
    
#     M,N = A.shape
#     H = algopy.zeros((M,M),dtype=A)
#     for n in range(N)[::-1]:
#         v,beta = house(A[n:,n:n+1])
#         A[n:,n:] -= beta * algopy.dot(v, algopy.dot(v.T,A[n:,n:]))
#         H[...] = numpy.eye(M)
#         H[n:,n:] -= beta * algopy.dot(v,v.T)
#         Q = algopy.dot(Q,H)
        
#     return Q, algopy.triu(A)
    


D,P,M,N = 50,1,3,2
A = algopy.UTPM(numpy.random.random((D,P,M,N)))
# Q,R = qr_house(A.copy())
# print algopy.dot(Q.T,Q) - numpy.eye(M)
# print algopy.dot(Q,R) - A

Q,R = algopy.qr_full(A.copy())
print algopy.dot(Q.T,Q) - numpy.eye(M)
print algopy.dot(Q,R) - A

        
        
