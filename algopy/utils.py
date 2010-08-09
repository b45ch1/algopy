import numpy
import numpy.testing

import globalfuncs

def utps2base_and_dirs(x):
    """
    Conversion function from algopy.utps to adolc.hov_forward input argument format.
    
    This function converts a 1D array of UTPS instances to the format (y,W)
    where x is an (Nx,)-array and W an (Nx,P,D-1) array.
    
    (y,W) is the input format for adolc.hov_forward
    D is the largest degree of the polynomial, i.e. t^D
    """
    Nx = numpy.shape(x)[0]
    D,P = numpy.shape(x[0].data)
    
    
    y = numpy.zeros(Nx,dtype=float)
    W = numpy.zeros((Nx,P,D-1))
    
    for nx in range(Nx):
        y[nx] = x[nx].data[0,0]
        W[nx,:,:] = x[nx].data[1:,:].T
        
    return (y,W)
    

def base_and_dirs2utps(x,V):
    """
    this function converts (x,V) to an Nx array of UTPS instances.
    where x is an (Nx,)-array and V an (Nx,P,D-1) array.
    (x,V) is the input format for adolc.hov_forward
    
    D is the largest degree of the polynomial, i.e. t^D
    """

    Nx = numpy.shape(x)[0]
    (Nx2,P,D) = numpy.shape(V)
    D += 1
    assert Nx == Nx2
    
    y = []
    for nx in range(Nx):
        tmp = numpy.zeros((D,P),dtype=float)
        tmp[0,:]  = x[nx]
        tmp[1:,:] = V[nx,:,:].T
        y.append(UTPS(tmp))
    return numpy.array(y)

def utpm2dirs(u):
    """
    Vbar = utpm2dirs(u)
    
    where u is an UTPM instance with
    u.data.shape = (D,P) + shp
    
    and  V.shape == shp + (P,D)
    """
    axes =  tuple( numpy.arange(2,u.data.ndim))+ (1,0)
    Vbar = u.data.transpose(axes)
    return Vbar


def utpm2base_and_dirs(u):
    """
    x,V = utpm2base_and_dirs(u)
    
    where u is an UTPM instance with
    u.data.shape = (D+1,P) + shp
    
    then x.shape == shp
    and  V.shape == shp + (P,D)
    """
    D,P = u.data.shape[:2]
    D -= 1
    shp = u.data.shape[2:]
    
    x = numpy.zeros(shp)
    V = numpy.zeros(shp+(P,D))
    
    x[...] = u.data[0,0,...]
    V[...] = u.data[1:,...].transpose( tuple(2+numpy.arange(len(shp))) + (1,0))
    return x,V

    
def base_and_dirs2utpm(x,V):
    """
    x_utpm = base_and_dirs2utpm(x,V)
    where x_utpm is an instance of UTPM
    V.shape = x.shape + (P,D)
    then x_utpm.data.shape = (D+1,P) = x.shape
    """
    x = numpy.asarray(x)
    V = numpy.asarray(V)
    
    xshp = x.shape
    Vshp = V.shape
    P,D = Vshp[-2:]
    Nxshp = len(xshp)
    NVshp = len(Vshp)
    numpy.testing.assert_array_equal(xshp, Vshp[:-2], err_msg = 'x.shape does not match V.shape')
    
    tc = numpy.zeros((D+1,P) + xshp)
    for p in range(P):
        tc[0,p,...] = x[...]
    
    axes_ids = tuple(numpy.arange(NVshp))    
    tc[1:,...] = V.transpose((axes_ids[-1],axes_ids[-2]) + axes_ids[:-2])
    
    return UTPM(tc)

def symvec(A):
    """ 
    
    returns the distinct elements of a symmetrized square matrix A as vector
    
    Example 1:
    ~~~~~~~~~~
        
        A = [[0,1,2],[1,3,4],[2,4,5]]
        v = symvec(A)
        returns v = [0,1,2,3,4,5]
        
    Example 2:
    ~~~~~~~~~~
        
        A = [[1,2],[3,4]]
        is not symmetric and symmetrized, yielding
        v = [1, (2+3)/2, 4]
        as output

    """
    
    N,M = A.shape
    
    assert N == M
    
    v = globalfuncs.zeros( ((N+1)*N)//2, dtype=A)
    
    count = 0
    for row in range(N):
        for col in range(row,N):
            v[count] = 0.5* (A[row,col] + A[col,row])
            count +=1
    return v

def vecsym(v):
    """
    returns a full symmetric matrix filled
    the distinct elements of v, filled row-wise
    """
    
    Nv = v.size
    N = (int(numpy.sqrt(1 + 8*Nv)) - 1)//2

    A = globalfuncs.zeros( (N,N), dtype=v)
    
    count = 0
    for row in range(N):
        for col in range(row,N):
            A[row,col] = A[col,row] = v[count]
            count +=1
    
    return A
    

