import numpy
import string
from algopy.utp.utpm import UTPM
from algopy.tracer import Function

# override numpy definitions

numpy_function_names = ['trace', 'dot', 'zeros_like']
numpy_linalg_function_names = ['inv', 'solve', 'eig', 'qr']




function_template = string.Template('''
def $function_name(*args):
    if isinstance(args[0], numpy.ndarray):
        return $namespace.__getattribute__('$function_name')(*args)
    elif hasattr(args[0].__class__, '$function_name'):
        return getattr(args[0].__class__, '$function_name')(*args)
    else:
        return $namespace.__getattribute__('$function_name')(*args)
''')

for function_name in numpy_function_names:
    exec function_template.substitute(function_name=function_name, namespace='numpy')
    
for function_name in numpy_linalg_function_names:
    exec function_template.substitute(function_name=function_name, namespace='numpy.linalg')


def zeros( shape, dtype=float, order = 'C'):
    """
    generalization of numpy.zeros
    
    create a zero instance
    """
    
    if isinstance(dtype,type):
        return numpy.zeros(shape, dtype=dtype,order=order)
    
    elif isinstance(dtype, numpy.ndarray):
        return numpy.zeros(shape,dtype=dtype.dtype, order=order)

    elif isinstance(dtype, UTPM):
        D,P = dtype.data.shape[:2]
        return dtype.__class__(numpy.zeros((D,P) + shape ,dtype = float))
        
    elif isinstance(dtype, Function):
        return dtype.__class__(zeros(shape, dtype=dtype.x, order = order))
        
    else:
        raise ValueError('don\'t know what to do with dtype = %s'%str(dtype))
