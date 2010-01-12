import numpy
import string
from algopy.utp.utpm import UTPM

# override numpy definitions

numpy_function_names = ['trace', 'dot']
numpy_linalg_function_names = ['inv', 'solve', 'eig']

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
    
# def shape(x):
    # if hasattr(x.__class__, 'shape'):
        # return x.__class__.shape(x)
    # else:
        # return numpy.shape(x)
        
# def size(x):
    # if isinstance(x, UTPM):
        # return x.size
    # else:
        # return numpy.size(x)
        
# def trace(x):
    # if isinstance(x, UTPM):
        # return x.trace()
    # else:
        # return numpy.trace(x)
        
# def inv(x):
    # if isinstance(x, UTPM):
        # return UTPM.inv(x)
    # else:
        # return numpy.linalg.inv(x)
        
# def dot(x,y, out = None):
    
    # if out != None:
        # raise NotImplementedError('should implement that...')
    
    # if isinstance(x, UTPM) or isinstance(y, UTPM):
        # return UTPM.dot(x,y)
        
    # else:
        # return numpy.dot(x,y)
        
        
# def solve(A,x):
    # if isinstance(x, UTPM):
        # raise NotImplementedError('should implement that...')
    
    # elif isinstance(A, UTPM):
        # raise NotImplementedError('should implement that...')
    
    # else:
        # return numpy.linalg.solve(A,x)
        
# def eig(A):
    # if isinstance(A, UTPM):
        # return UTPM.eig(A)
    
    # else:
        # return numpy.linalg.eig(A)





def combine_blocks(in_X):
    """
    expects an array or list consisting of entries of type UTPM, e.g.
    in_X = [[UTPM1,UTPM2],[UTPM3,UTPM4]]
    and returns
    UTPM([[UTPM1.data,UTPM2.data],[UTPM3.data,UTPM4.data]])

    """

    in_X = numpy.array(in_X)
    Rb,Cb = numpy.shape(in_X)

    # find the degree D and number of directions P
    D = 0; 	P = 0;

    for r in range(Rb):
        for c in range(Cb):
            D = max(D, in_X[r,c].data.shape[0])
            P = max(P, in_X[r,c].data.shape[1])

    # find the sizes of the blocks
    rows = []
    cols = []
    for r in range(Rb):
        rows.append(in_X[r,0].shape[0])
    for c in range(Cb):
        cols.append(in_X[0,c].shape[1])
    rowsums = numpy.array([ numpy.sum(rows[:r]) for r in range(0,Rb+1)],dtype=int)
    colsums = numpy.array([ numpy.sum(cols[:c]) for c in range(0,Cb+1)],dtype=int)

    # create new matrix where the blocks will be copied into
    tc = numpy.zeros((D, P, rowsums[-1],colsums[-1]))
    for r in range(Rb):
        for c in range(Cb):
            tc[:,:,rowsums[r]:rowsums[r+1], colsums[c]:colsums[c+1]] = in_X[r,c].data[:,:,:,:]

    return UTPM(tc) 
