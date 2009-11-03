"""
This implements a class factory that returns classes that share a common interface. 

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

class GradedRing(object):
    data = NotImplementedError()
    
    def __add__(self, rhs):
        retval = self.__class__(self.__class__.zeros_like(self.data))
        self.__class__.add(retval.data, self.data, rhs.data)
        return retval
        
    def __sub__(self, rhs):
        retval = self.__class__(self.__class__.zeros_like(self.data))
        self.__class__.sub(retval.data, self.data, rhs.data)
        return retval        

    def __mul__(self,rhs):
        retval = self.__class__(self.__class__.zeros_like(self.data))
        self.__class__.mul(retval.data, self.data, rhs.data)
        return retval
        
    def __div__(self,rhs):
        retval = self.__class__(self.__class__.zeros_like(self.data))
        self.__class__.div(retval.data, self.data, rhs.data)
        return retval        
        
    def __radd__(self, lhs):
        return self + lhs
        
    def __rmul__(self, lhs):
        return self * lhs
        
    def __str__(self):
       return str(self.data)
 


