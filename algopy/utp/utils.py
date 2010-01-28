import numpy
import numpy.testing
from algopy.utp.utps import UTPS
from algopy.utp.utpm import UTPM


def utpm2utps(x):
    """
    converts an instance of UTPM with  x.data.shape = P,D,N,M
    to a (N,M) array of UTPS instances y_ij, where y_ij.data.shape = (P,D)
    """
    P,D,N,M = x.data.shape
    
    tmp_n = []
    for n in range(N):
        tmp_m = []
        for m in range(M):
            tmp_m.append( UTPS(x.data[:,:,n,m]))
        tmp_n.append(tmp_m)
    
    return numpy.array(tmp_n)
    
    
def utps2utpm(x):
    """
    converts a 2D array x of UTPS instances with x.shape = (N,M)
    and x_ij.data.shape = (D,P)
    to a UTPM instance y where y.data.shape = (D,P,N,M)
    
    if x is a 1D array it is converted to a (D,P,N,1) matrix
    
    """
    
    if numpy.ndim(x) == 1:
        x = numpy.reshape(x, (numpy.size(x),1))
    
    N,M = numpy.shape(x)
    P,D = numpy.shape(x[0,0].data)
    
    tmp = numpy.zeros((P,D,N,M),dtype=float)
    
    for n in range(N):
        for m in range(M):
            tmp[:,:,n,m] = x[n,m].data[:,:]
    
    return UTPM(tmp)

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

