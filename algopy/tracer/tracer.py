import numpy

from algopy.base_type import Algebra

class CGraph:
    """
    The CGraph (short for Computational Graph) represents a computational
    procedure of elementary functions.
    It is a Directed Acylic Graph (DAG).
    
    The data structure is a list of vertices and each vertex knows its parents.
    
    Example:
    
        The Graph of y = x1(x2+x3) looks like
    
        OP    ID ARGS     VALUE
        -----------------------------------------------------------
        Id     1 [1]         x1            set independent variable
        Id     2 [2]         x2            set independent variable
        Id     3 [3]         x3            set independent variable        
        +      4 [v2,v3]  (v2.x + v3.x)
        *      5 [v1,+4]  (v1.x * v4.x)
        
        --- independent variables
        [1,2,3]
        
        --- dependent variables
        [5]
        
        
    One can perform to actions with a computational graph:
    
        1)  push forward
        2)  pullback of the differential of the dependent variables
        
    The push forward can propagate any data structure.
    For instance:
        * univariate Taylor polynomials over scalars (e.g. implemented in algopy.utp.utps)
        * univariate Taylor polynomials over matrices (e.g. implemented in algopy.utp.utpm)
        * cross Taylor polynomials over scalars (e.g. implemented in algopy.utp.ctps_c)
        * multivariate Taylor polynomials (not implemented)
        
    Relaxing the definition of the push forward one could also supply:
        * intervals arithmetic
        * slopes
        * etc.
        
        
    The pullback is specifically the pullback of an element of the cotangent space that linearizes
    the computational procedure.
        
    
    """

    def __init__(self):
        self.functionCount = 0
        self.functionList = []
        self.dependentFunctionList = []
        self.independentFunctionList = []
        Function.cgraph = self
        
    def append(self, func):
        self.functionCount += 1
        self.functionList.append(func)
        
    def __str__(self):
        retval = ''
        for f in self.functionList:
            if numpy.size(f.args)>1:
                arg_IDS = [ af.ID for af in f.args]
            else:
                arg_IDS = f.args.ID
                
            retval += '%s: IDs: %s <- %s  Values: %s <-  %s\n'%(str(f.func.__name__), str(f.ID), str(arg_IDS), str(f.x),str(f.args))
        return retval
        
    def push_forward(self,x_list):
        """
        Apply a global push forward of the computational procedure defined by
        the computational graph saved in this CGraph instance.
        
        At first, the arguments of the global functions are read into the independent functions.
        Then the computational graph is walked and at each function node
        """
        # populate independent arguments with new values
        for nf,f in enumerate(self.independentFunctionList):
            f.args.x = x_list[nf]

        # traverse the computational tree
        for f in self.functionList:
            f.__class__.push_forward(f.func, f.args, Fout = f)


class Function(Algebra):
    
    def __init__(self, x = None):
        """
        Creates a new function node that is a variable.
        """
        
        if x != None:
            # create a Function node with value x referring to itself, i.e.
            # returning x when called
            cls = self.__class__
            cls.create(x, self, cls.Id, self)    
    
    cgraph = None
    @classmethod
    def get_ID(cls):
        """
        return function ID
        
        Rationale:
            When code tracing is active, each Function is added to a Cgraph instance
            and it is given a unique ID.
        
        """
        if cls.cgraph != None:
            return cls.cgraph.functionCount
        else:
            return None
    
    @classmethod
    def create(cls, x, args, func, f = None):
        """
        Creates a new function node.
        
        INPUTS:
            x           anything                            current value
            args        tuple of Function objects           arguments of the new Function node
            func        callable                            the function that can evaluate func(x)
            
        OPTIONAL:
            f           Function instance
        
        """
        if f == None:
            f = Function()
        f.x = x
        f.args = args
        f.func = func
        f.ID = cls.get_ID()
        cls.cgraph.append(f)
        return f
    
    @classmethod    
    def Id(cls, x):
        """
        The identity function:  x = Id(x)
        
        """
        return x


    def __repr__(self):
        return str(self)

    def __str__(self):
        return '%s'%str(self.x)

        
    @classmethod
    def push_forward(cls, func, Fargs, Fout = None):
        """
        Computes the push forward of func
        
        INPUTS:
            func            callable            func(Fargs[0].x,...,Fargs[-1].x)
            Fargs           tuple               tuple of Function nodes
        """
        if numpy.ndim(Fargs) > 0:
            args = tuple([ fa.x for fa in Fargs])
            out  = func(*args)
        else:
            arg = Fargs.x
            out  = func(arg)
        
        if Fout == None:
            return cls.create(out, Fargs, func)
        
        else:
            Fout.x = out
            return Fout
            
            
            
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
            self.__class__.__scalar_to_data__(xdata, x)
            return self.__class__(xdata)
            
        elif isinstance(x, numpy.ndarray):
            raise NotImplementedError('sorry, not implemented just yet')
            
        elif not isinstance(x, self.__class__):
            raise NotImplementedError('Cannot convert x\n type(x) = %s but expected type(x) = %s'%(str(type(x))))
        
        else:
            return x            
    
    def __add__(self,rhs):
        return Function.push_forward(self.x.__class__.__add__,(self,rhs))

    def __sub__(self,rhs):
        return Function.push_forward(self.x.__class__.__sub__,(self,rhs))

    def __mul__(self,rhs):
        return Function.push_forward(self.x.__class__.__mul__,(self,rhs))

    def __div__(self,rhs):
        return Function.push_forward(self.x.__class__.__div__,(self,rhs))



        
