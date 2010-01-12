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
        retval = ''
        for f in self.functionList:
            if numpy.size(f.args)>1:
                arg_IDS = [ af.ID for af in f.args]
            else:
                arg_IDS = f.args.ID
                
            retval += '%s: IDs: %s <- %s  Values: %s <-  %s\n'%(str(f.func.__name__), str(f.ID), str(arg_IDS), str(f.x),str(f.args))
        return retval
        
    def push_forward(self,x_list):
        # populate independent arguments with new values
        for nf,f in enumerate(self.independentFunctionList):
            f.args.x = x_list[nf]

        # traverse the computational tree
        for f in self.functionList:
            f.__class__.push_forward(f.func, f.args, Fout = f)


class Function:
    
    def __init__(self, x = None):
        """
        Creates a new function node that is an independent variable.
        """
        
        if x != None:
            cls = self.__class__
            cls.create(x, self, cls.Id, self)    
    
    
    
    cgraph = None
    @classmethod
    def get_ID(cls):
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
         
        
    
    def pullback():
        pass
