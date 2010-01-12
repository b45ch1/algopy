import numpy

class CG:
    """
    We implement the Computational Graph (CG) as Directed Acyclic Graph.
    The Graph of y = x1(x2+x3) looks like

    --- independent variables
    v1(x1): None
    v2(x2): None
    v3(x3): None

    --- function operations
    +4(v2.x + v3.x): [v2,v3]
    *5(v1.x * +4.x): [v1,+4]

    --- dependent variables
    v6(*5.x): [*5]
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
        return 'vertices:\n' + str(self.functionList)

    def push_forward(self,x_list):
        # populate independent arguments with new values
        for nf,f in enumerate(self.independentFunctionList):
            f.args = x_list[nf]

        # traverse the computational tree
        for f in self.functionList:
            f.x = f.__class__.push_forward(f.func, f.args)


class Function:
    
    def __init__(self, x = None):
        """
        Creates a new function node that is an independent variable.
        """
        self.x = x
        self.func = self.id
        self.args = None
        
    def id(self):
        return self.x

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '%s(f)'%str(self.x)
    
    @classmethod
    def create(cls, x, args, func):
        """
        Creates a new function node.
        
        INPUTS:
            x           anything                            current value
            args        tuple of Function objects           arguments of the new Function node
            func        callable                            the function that can evaluate func(x)
        
        """
        
        f = Function()
        f.x = x
        f.args = args
        f.func = func
        cls.cgraph.append(f)
        return f
        
    @classmethod
    def push_forward(cls, func, Fargs):
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
        return cls.create(out, Fargs, func)
        
    
    def pullback():
        pass
