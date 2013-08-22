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

def qr_house_basic(A):
    """ computes QR decomposition using Householder relections
    qr_house_basic and build_Q have the same effect as qr_house
    
    Parameters
    ----------
    A: array_like
       shape(A) = (M, N), M >= N
       overwritten on exit
    
    Returns
    -------
    A: array_like
        strict lower triangular part contains the Householder vectors v
        upper triangular matrix R
    
    betas: array_like
        2-norms of the Householder vectors v
    """
    
    M,N = A.shape
    beta_list = []
    for n in range(N):
        v,beta = house(A[n:,n:n+1])
        A[n:,n:] -= beta * algopy.dot(v, algopy.dot(v.T,A[n:,n:]))
        
        beta_list.append(beta)
        if n < M:
            A[n+1:,n] = v[1:,0]
    return A, numpy.asarray(beta_list)
    
def build_Q(A, betas):
    """ computes orthogonal matrix from output of qr_house_basic
    
    Parameters
    ----------
    A: array_likse
        shape(A) = (M,N)
        upper triangular part contains R
        lower triangular part contains v with v[0] = 1
    betas: array_like
        list of beta
        
    Returns
    -------
    Q: array_like
        shape(Q) = (M,M)

    """
    
    M,N = A.shape
    Q = algopy.zeros((M,M),dtype=A)
    Q += numpy.eye(M)
    H = algopy.zeros((M,M),dtype=A)
    for n in range(N):
        v = A[n:,n:n+1].copy()
        v[0] = 1
        H[...] = numpy.eye(M)
        H[n:,n:] -= betas[n] * algopy.dot(v,v.T)
        Q = algopy.dot(Q,H)
    return Q
    
    
def pb_qr_house(A, Abar, Q, Qbar, R, Rbar):
    """ computes the pullback of qr_house
       
    Parameters
    ----------
    A: array_like
       shape(A) = (M, N), M >= N

    Abar: array_like
       shape(Abar) = (M, N), M >= N
       changed on exit

    Q: array_like
       shape(Q) = (M, M)
       
    Qbar: array_like
       shape(Qbar) = (M, M)
       changed on exit
       
    R: array_like
       shape(R) = (M, N)       
       
    Rbar: array_like
       shape(R) = (M, N)
       changed on exit

    """
    
    raise NotImplementedError('')
    

import time

D,P,M,N = 50,1,3,2
# # STEP 1: qr_house_basic + build_Q
# A = numpy.random.random((M,N))
# B,betas = qr_house_basic(A.copy())
# R = algopy.triu(B)
# Q = build_Q(B,betas)
# print algopy.dot(Q.T,Q) - numpy.eye(M)
# print algopy.dot(Q,R) - A

# # STEP 2: qr_house
# Q,R = qr_house(A.copy())
# print algopy.dot(Q.T,Q) - numpy.eye(M)
# print algopy.dot(Q,R) - A

# # STEP 3: qr_full
# Q,R = algopy.qr_full(A.copy())
# print algopy.dot(Q.T,Q) - numpy.eye(M)
# print algopy.dot(Q,R) - A


# data = numpy.random.random((D,P,M,N))
# data = numpy.asarray(data, dtype=numpy.float64)
# A = algopy.UTPM(data)
# # STEP 1: qr_house_basic + build_Q
# print 'QR decomposition based on basic Householder'
# st = time.time()
# B,betas = qr_house_basic(A.copy())
# R = algopy.triu(B)
# Q = build_Q(B,betas)
# # print algopy.dot(Q.T,Q) - numpy.eye(M)
# # print algopy.dot(Q,R) - A
# print 'runtime = ',time.time() - st

# # STEP 2: qr_house
# print 'QR decomposition based on Householder'
# st = time.time()
# Q2,R2 = qr_house(A.copy())
# print algopy.dot(Q2.T,Q2) - numpy.eye(M)
# print algopy.dot(Q2,R2) - A
# print 'runtime = ',time.time() - st

# # STEP 3: qr_full
# print 'QR decomposition based on defining equations'
# st = time.time()

# Q,R = algopy.qr_full(A.copy())
# print algopy.dot(Q.T,Q) - numpy.eye(M)
# print algopy.dot(Q,R) - A
# print 'runtime = ',time.time() - st


# SAVE matrices in .mat format for Rene Lamour (who uses Matlab...)
# -----------------------------------------------------------------
# import scipy.io
# scipy.io.savemat('matrix_polynomial.mat', {"A_coeffs": A.data,
# "Q_coeffs_house":Q2.data,
# "R_coeffs_house":R2.data,
# "defect_QR_minus_A_house": (algopy.dot(Q2,R2) - A).data,
# "defect_QTQ_minus_Id_house": (algopy.dot(Q2.T,Q2) - numpy.eye(M)).data,
# "Q_coeffs":Q.data,
# "R_coeffs":R.data,
# "defect_QR_minus_A": (algopy.dot(Q,R) - A).data,
# "defect_QTQ_minus_Id": (algopy.dot(Q.T,Q) - numpy.eye(M)).data
# })


import mpmath
mpmath.mp.prec = 200 # increase lenght of mantissa
print(mpmath.mp)

print('QR decomposition based on Householder')
D,P,M,N = 50,1,3,2

# in float64 arithmetic
data = numpy.random.random((D,P,M,N))
data = numpy.asarray(data, dtype=numpy.float64)
A = algopy.UTPM(data)
Q,R = qr_house(A.copy())

# in multiprecision arithmetic
data2 = numpy.asarray(data)*mpmath.mpf(1)
A2 = algopy.UTPM(data2)
Q2,R2 = qr_house(A2.copy())

print('-'*20)
print(A.data[-1])
print('-'*20)
print((Q.data[-1] - Q2.data[-1])/Q2.data[-1])
print('-'*20)
print(algopy.dot(Q, R).data[-1] - A.data[-1])
print('-'*20)
print(algopy.dot(Q2, R2).data[-1] - A2.data[-1])


# # print algopy.dot(Q2.T,Q2) - numpy.eye(M)
# # print algopy.dot(Q2,R2) - A
        
