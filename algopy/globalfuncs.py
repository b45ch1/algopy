import math
import numpy
try:
    import scipy;
    import scipy.linalg
    import scipy.special

except:
    pass

import string
import utils
from algopy import UTPM
from algopy import Function

# override numpy definitions
numpy_function_names = ['sin','cos','tan', 'exp', 'log', 'sqrt', 'pow', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh', 'trace',  'zeros_like', 'diag', 'triu', 'tril', 'reshape']
numpy_linalg_function_names = ['inv', 'solve', 'eigh', 'qr', 'cholesky','transpose', 'det']


function_template = string.Template('''
def $function_name(*args, **kwargs):
    """
    generic implementation of $function_name

    this function calls, depending on the input arguments,
    either

    * numpy.$function_name
    * numpy.linalg.$function_name
    * args[i].__class__

    """
    case,arg = 0,0
    for na,a in enumerate(args):
        if hasattr(a.__class__, '$function_name'):
            case = 1
            arg  = na
            break

    if case==1:
        return getattr(args[arg].__class__, '$function_name')(*args, **kwargs)

    elif case==0:
        return $namespace.__getattribute__('$function_name')(*args, **kwargs)

    else:
        return $namespace.__getattribute__('$function_name')(*args, **kwargs)
''')

for function_name in numpy_function_names:
    exec function_template.substitute(function_name=function_name, namespace='numpy')

for function_name in numpy_linalg_function_names:
    exec function_template.substitute(function_name=function_name, namespace='numpy.linalg')

def sum(x, axis=None, dtype=None, out=None):
    """ generic sum function
    calls either numpy.sum or Function.sum resp. UTPM.sum depending on
    the input
    """

    if isinstance(x, numpy.ndarray) or numpy.isscalar(x):
        return numpy.sum(x, axis=axis, dtype=dtype, out = out)

    elif isinstance(x, UTPM) or isinstance(x, Function):
       return x.sum(axis = axis, dtype = dtype, out = out)

    else:
        raise ValueError('don\'t know what to do with this input!')


def prod(x, axis=None, dtype=None, out=None):
    """ generic prod function
    """

    if axis != None or dtype != None or out != None:
        raise NotImplementedError('')

    elif isinstance(x, numpy.ndarray):
        return numpy.prod(x)

    elif isinstance(x, Function) or  isinstance(x, UTPM):
        y = zeros(1,dtype=x)
        y[0] = x[0]
        for xi in x[1:]:
            y[0] = y[0] * xi
        return y[0]


def coeff_op(x, sl, shp):
    return x.coeff_op(sl, shp)



def init_UTPM_jacobian(x):
    # print 'type(x)=', type(x)
    if isinstance(x, Function):
        return x.init_UTPM_jacobian()

    elif isinstance(x, numpy.ndarray):
        return UTPM.init_jacobian(x)

    elif isinstance(x, UTPM):
        # print x.data.shape
        return UTPM.init_UTPM_jacobian(x.data[0,0])

    else:
        raise ValueError('don\'t know what to do with this input!')

def extract_UTPM_jacobian(x):
    if isinstance(x, Function):
        return x.extract_UTPM_jacobian()

    elif isinstance(x, UTPM):
        return UTPM.extract_UTPM_jacobian(x)
    else:
        raise ValueError('don\'t know what to do with this input!')



def zeros( shape, dtype=float, order = 'C'):
    """
    generic generalization of numpy.zeros

    create a zero instance
    """

    if numpy.isscalar(shape):
        shape = (shape,)

    if isinstance(dtype,type):
        return numpy.zeros(shape, dtype=dtype,order=order)

    elif isinstance(dtype, numpy.ndarray):
        return numpy.zeros(shape,dtype=dtype.dtype, order=order)

    elif isinstance(dtype, UTPM):
        D,P = dtype.data.shape[:2]
        tmp = numpy.zeros((D,P) + shape ,dtype = dtype.data.dtype)
        tmp*= dtype.data.flatten()[0]
        return dtype.__class__(tmp)

    elif isinstance(dtype, Function):
        # dtype.create(zeros(shape, dtype=dtype.x, order = order), fargs, zeros):
        return dtype.pushforward(zeros, [shape, dtype, order])
        # return dtype.__class__(zeros(shape, dtype=dtype.x, order = order))

    else:
        return numpy.zeros(shape,dtype=type(dtype), order=order)
        # raise ValueError('don\'t know what to do with dtype = %s, type(dtype)=%s'%(str(dtype), str(type(dtype))))

def dot(a,b):
    """
    Same as NumPy dot but in UTP arithmetic
    """
    if isinstance(a,Function) or isinstance(b,Function):
        return Function.dot(a,b)

    elif isinstance(a,UTPM) or isinstance(b,UTPM):
        return UTPM.dot(a,b)

    else:
        return numpy.dot(a,b)

def outer(a,b):
    """
    Same as NumPy outer but in UTP arithmetic
    """
    if isinstance(a,Function) or isinstance(b,Function):
        return Function.outer(a,b)

    elif isinstance(a,UTPM) or isinstance(b,UTPM):
        return UTPM.outer(a,b)

    else:
        return numpy.outer(a,b)



def qr_full(A):
    """
    Q,R = qr_full(A)

    This function is merely a wrapper of
    UTPM.qr_full,  Function.qr_full, scipy.linalg.qr

    Parameters
    ----------

    A:      algopy.UTPM or algopy.Function or numpy.ndarray
            A.shape = (M,N),  M >= N

    Returns
    --------

    Q:      same type as A
            Q.shape = (M,M)

    R:      same type as A
            R.shape = (M,N)


    """

    if isinstance(A, UTPM):
        return UTPM.qr_full(A)

    elif isinstance(A, Function):
        return Function.qr_full(A)

    elif isinstance(A, numpy.ndarray):
        return scipy.linalg.qr(A)

    else:
        raise NotImplementedError('don\'t know what to do with this instance')


def eigh1(A):
    if isinstance(A, UTPM):
        return UTPM.eigh1(A)

    elif isinstance(A, Function):
        return Function.eigh1(A)

    elif isinstance(A, numpy.ndarray):
        A = UTPM(A.reshape((1,1) + A.shape))
        retval = UTPM.eigh1(A)
        return retval[0].data[0,0], retval[1].data[0,0],retval[2]

    else:
        raise NotImplementedError('don\'t know what to do with this instance')


def symvec(A, UPLO='F'):
    if isinstance(A, UTPM):
        return UTPM.symvec(A, UPLO=UPLO)

    elif isinstance(A, Function):
        return Function.symvec(A, UPLO=UPLO)

    elif isinstance(A, numpy.ndarray):
        return utils.symvec(A, UPLO=UPLO)

    else:
        raise NotImplementedError('don\'t know what to do with this instance')
symvec.__doc__ = utils.symvec.__doc__


def vecsym(v):
    if isinstance(v, UTPM):
        return UTPM.vecsym(v)

    elif isinstance(v, Function):
        return Function.vecsym(v)

    elif isinstance(v, numpy.ndarray):
        return utils.vecsym(v)

    else:
        raise NotImplementedError('don\'t know what to do with this instance')
vecsym.__doc__ = utils.vecsym.__doc__


def svd(A, epsilon=10**-8):
    """
    computes the singular value decomposition A = U S V.T
    of matrices A with full rank (i.e. nonzero singular values)

    (U, S, VT) = UTPM.svd(A, epsilon= 10**-8)

    Parameters
    ----------

    A: array_like
        input array (numpy.ndarray, algopy.UTPM or algopy.Function instance)

    epsilon:   float
        threshold to evaluate the rank of A

    Implementation
    --------------

    The singular value decomposition is directly related to the symmetric
    eigenvalue decomposition.

    See A. Bjoerk, Numerical Methods for Least Squares Problems, SIAM, 1996
    for the relation between SVD and symm. eigenvalue decomposition

    and S. F. Walter, Structured Higher-Order Algorithmic Differentiation
    in the Forward and Reverse Mode with Application in Optimum Experimental
    Design, PhD thesis, 2011
    for the Taylor polynomial arithmetic.

    """

    M,N = A.shape

    if N > M:
        raise NotImplementedError("A.shape = (M,N) and N > M is not supported (yet)")

    # real symmetric eigenvalue decomposition

    B = zeros((M+N, M+N),dtype=A)
    B[:M,M:] = A
    B[M:,:M] = A.T
    l,Q = eigh(B, epsilon=epsilon)


    # compute the rank
    r = 0
    for i in range(N):
        if abs(l[i]) > epsilon:
            r = i+1

    if r < N:
        raise NotImplementedError('rank deficient matrices are not supported')

    # permutation matrix

    tmp  = [M+N-1 - i for i in range(r)] + [i for i in range(r)]
    tmp += [M+N-1-r-i for i in range(N-r)] + [N-1+i for i in range(N-r)]
    tmp += [N+i for i in range(M-N)]
    P = numpy.eye(M+N)
    P = P[tmp]


    # bring Q into the required format

    Q = dot(Q, P.T)


    # find U S V.T

    U = zeros((M,M), dtype=Q)
    V = zeros((N,N), dtype=Q)

    U[:,:r] = 2.**0.5*Q[:M,:r]
    U[:,r:] = Q[:M, 2*r: r+M]
    V[:,:r] = 2.**0.5*Q[M:,:r]
    V[:,r:] = Q[M:,r+M:]
    s = -l[:N]

    return U, s, V


def expm(A):
    """
    B = expm(A)

    Compute the matrix exponential using a Pade approximation of order 7.

    Warning
    -------

    A Pade approximation order 7 may no be sufficient to
    obtain derivatives that are accurate up to machine precision.

    Parameters
    ----------

    A:      array_like (algopy.UTPM, algopy.Function, numpy.ndarray)
            A.shape = (N,N)

    Returns
    -------

    B:      same type as A
            B.shape = (N,N)


    Reference
    ---------

    N. J. Higham,
    "The Scaling and Squaring Method for the Matrix Exponential Revisited",
    SIAM. J. Matrix Anal. & Appl. 26, 1179 (2005).
    """

    return expm_pade(A, 7)


def expm_pade(A, q):
    """
    Compute the matrix exponential using a fixed-order Pade approximation.
    """
    q_to_pade = {
            3 : _expm_pade3,
            5 : _expm_pade5,
            7 : _expm_pade7,
            9 : _expm_pade9,
            13 : _expm_pade13,
            }
    pade = q_to_pade[q]
    ident = numpy.eye(A.shape[0])
    U, V = pade(A, ident)
    return solve(-U + V, U + V)

def expm_higham_2005(A):
    """
    Compute the matrix exponential using the method of Higham 2005.

    N. J. Higham,
    "The Scaling and Squaring Method for the Matrix Exponential Revisited",
    SIAM. J. Matrix Anal. & Appl. 26, 1179 (2005).
    """
    n_squarings = 0
    # FIXME: is there an algopy norm implementation?
    A_L1 = numpy.linalg.norm(A, 1)
    ident = numpy.eye(A.shape[0])
    if A_L1 < 1.495585217958292e-002:
        U,V = _expm_pade3(A, ident)
    elif A_L1 < 2.539398330063230e-001:
        U,V = _expm_pade5(A, ident)
    elif A_L1 < 9.504178996162932e-001:
        U,V = _expm_pade7(A, ident)
    elif A_L1 < 2.097847961257068e+000:
        U,V = _expm_pade9(A, ident)
    else:
        maxnorm = 5.371920351148152
        # FIXME: this should probably use algopy log,
        #        and algopy max and ceil if they exist.
        n_squarings = max(0, int(math.ceil(math.log(A_L1 / maxnorm, 2))))
        A /= 2**n_squarings
        U, V = _expm_pade13(A, ident)
    R = solve(-U + V, U + V)
    for i in range(n_squarings):
        R = dot(R, R)
    return R

def _expm_pade3(A, ident):
    """ Helper function for Pade approximation of expm.
    """
    b = (120., 60., 12., 1.)
    A2 = dot(A, A)
    U = dot(A, b[3]*A2 + b[1]*ident)
    V = b[2]*A2 + b[0]*ident
    return U,V

def _expm_pade5(A, ident):
    """ Helper function for Pade approximation of expm.
    """
    b = (30240., 15120., 3360., 420., 30., 1.)
    A2 = dot(A, A)
    A4 = dot(A2, A2)
    U = dot(A, b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[4]*A4 + b[2]*A2 + b[0]*ident
    return U,V

def _expm_pade7(A, ident):
    """ Helper function for Pade approximation of expm.
    """
    b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
    A2 = dot(A, A)
    A4 = dot(A2, A2)
    A6 = dot(A2, A4)
    U = dot(A, b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U, V

def _expm_pade9(A, ident):
    """ Helper function for Pade approximation of expm.
    """
    b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
            2162160., 110880., 3960., 90., 1.)
    A2 = dot(A, A)
    A4 = dot(A2, A2)
    A6 = dot(A2, A4)
    A8 = dot(A2, A6)
    U = dot(A, b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[8]*A8 + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U,V

def _expm_pade13(A, ident):
    """ Helper function for Pade approximation of expm.
    """
    b = (
            64764752532480000., 32382376266240000., 7771770303897600.,
            1187353796428800., 129060195264000., 10559470521600.,
            670442572800., 33522128640., 1323241920.,
            40840800., 960960., 16380., 182., 1.)
    A2 = dot(A, A)
    A4 = dot(A2, A2)
    A6 = dot(A2, A4)
    U = dot(A,
            dot(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) + (
                b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident))
    V = dot(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) + (
            b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident)
    return U, V

