import numpy
import scipy; import scipy.linalg
import string
import utils
from algopy import UTPM
from algopy import Function

# override numpy definitions

numpy_function_names = ['sin','cos','tan', 'exp', 'log', 'sqrt', 'pow', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh', 'trace',  'zeros_like', 'diag', 'triu', 'tril']
numpy_linalg_function_names = ['inv', 'solve', 'eigh', 'qr', 'cholesky','transpose', 'det']


function_template = string.Template('''
def $function_name(*args):
    case,arg = 0,0
    for na,a in enumerate(args):
        if hasattr(a.__class__, '$function_name'):
            case = 1
            arg  = na
            break
            
    if case==1:
        return getattr(args[arg].__class__, '$function_name')(*args)

    elif case==0:
        return $namespace.__getattribute__('$function_name')(*args)

    else:
        return $namespace.__getattribute__('$function_name')(*args)
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

        return dtype.__class__(numpy.zeros((D,P) + shape ,dtype = dtype.data.dtype))
        
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
    


def qr_full(A):
    """ Q,R = qr_full(A) returns QR decomposition with quadratic Q 
    calls internally UTPM.qr_full or Function.qr_full
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


