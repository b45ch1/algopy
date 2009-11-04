"""
This implements an abstrace base class GradedRing .

Rationale:
    
    Goal is to separate the datatype specification from the algorithms and containers for the following reasons:
    
        1) It allows to directly use the algorithms *without* overhead. E.g. calling mul(z.data, x.data, y.data)
           has much less overhead than z = x.__mul__(y). data is to be kept as close as possible to
           machine primitives. E.g. data is array or tuple of arrays.
        2) Potential reuse of an algorithm in several datatypes.
        3) Relatively easy to connect high performance algorithms with a very highlevel abstract description.
           For instance, most programming languages allow calling C-functions. Therefore, the algorithms
           should be given as void fcn(int A, double B, ...)
        
    For instance, the datatype is a truncated Taylor polynomial R[t]/<t^D> of the class Foo.
    The underlying container is a simple array of doubles.

"""

import numpy

class GradedRing(object):
    """
    data has to be mutable because it is passed as reference to algorithms, e.g. as
    add(result.data, lhs.data, rhs.data) where add changed result.data inplace.
    """
    data = NotImplementedError()
    
    def totype(self, x):
        """
        tries to convert x to an object of the class
        
        works for : scalar x, numpy.ndarray x
        
        Remark:
            at the moment, scalar x expanded as GradedRing with the same degree as self though. 
            The reason is a missing implementation that works for graded rings of different degree.
            Once such implementations exist, this function should be adapted.
        
        """
        if numpy.isscalar(x):
            xdata = self.__class__.__zeros_like__(self.data)
            self.__class__.__scalar_to_data__(self.__class__, x, xdata)
            return self.__class__(xdata)
            
        elif isinstance(x, numpy.ndarray):
            raise NotImplementedError('sorry, not implemented just yet')
            
        elif not isinstance(x, self.__class__):
            raise NotImplementedError('Cannot convert x\n type(x) = %s but expected type(x) = %s'%(str(type(x))))
        
        else:
            return x

    
    def __add__(self, rhs):
        rhs = self.totype(rhs)
        retval = self.__class__(self.__class__.__zeros_like__(self.data))
        self.__class__.add(retval.data, self.data, rhs.data)
        return retval
        
    def __sub__(self, rhs):
        rhs = self.totype(rhs)
        retval = self.__class__(self.__class__.__zeros_like__(self.data))
        self.__class__.sub(retval.data, self.data, rhs.data)
        return retval        

    def __mul__(self,rhs):
        rhs = self.totype(rhs)
        retval = self.__class__(self.__class__.__zeros_like__(self.data))
        self.__class__.mul(retval.data, self.data, rhs.data)
        return retval
        
    def __div__(self,rhs):
        rhs = self.totype(rhs)
        retval = self.__class__(self.__class__.__zeros_like__(self.data))
        self.__class__.div(retval.data, self.data, rhs.data)
        return retval        
        
    def __radd__(self, lhs):
        return self + lhs
        
    def __rmul__(self, lhs):
        return self * lhs
        
    def zeros_like(self):
        return self.__class__(self.__class__.__zeros_like__(self.data))
        
    def __str__(self):
       return str(self.data)
 


