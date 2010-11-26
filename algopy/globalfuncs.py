import numpy
import scipy; import scipy.linalg
import string
import utils
from algopy import UTPM
from algopy import Function

# override numpy definitions

numpy_function_names = ['sin','cos','tan', 'exp', 'log', 'sqrt', 'pow', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh', 'trace', 'dot', 'zeros_like', 'diag', 'triu', 'tril','sum']
numpy_linalg_function_names = ['inv', 'solve', 'eigh', 'qr', 'cholesky','transpose']


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
        raise ValueError('don\'t know what to do with dtype = %s, type(dtype)=%s'%(str(dtype), str(type(dtype))))
        

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
        
    else:
        raise NotImplementedError('don\'t know what to do with this instance')
        
def symvec(A):
    if isinstance(A, UTPM):
        return UTPM.symvec(A)
    
    elif isinstance(A, Function):
        return Function.symvec(A)
        
    elif isinstance(A, numpy.ndarray):
        return utils.symvec(A)
        
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

