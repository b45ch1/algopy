import numpy

from algopy.base_type import Ring

class NotSet:
    def __init__(self, descr=None):
        if descr == None:
            descr = ''
        self.descr = descr
    def __str__(self):
        return 'not set!'
        
def is_set(o):
    return not isinstance(o, NotSet)
        
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
        +      4 [2,3]   (v2.x + v3.x)
        *      5 [1,4]   (v1.x * v4.x)
        
        --- independent variables
        [1,2,3]
        
        --- dependent variables
        [5]
        
        
    One can perform to actions with a computational graph:
    
        1)  push forward
        2)  pullback of the differential of the dependent variables
        
    The push forward can propagate any data structure.
    For instance:
        * univariate Taylor polynomials over scalars (e.g. implemented in algopy.utps)
        * univariate Taylor polynomials over matrices (e.g. implemented in algopy.utpm)
        * cross Taylor polynomials over scalars (e.g. implemented in algopy.ctps_c)
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
        retval = '\n\n'
        for f in self.functionList:
            arg_IDs = []
            for a in f.args:
                if isinstance(a, Function):
                    arg_IDs.append(a.ID)
                else:
                    arg_IDs.append('c(%s)'%str(a))
            retval += '\n\n%s: IDs: %s <- %s\n'%(str(f.func.__name__), str(f.ID), str(arg_IDs))
            retval += 'x:\n    %s \n'%( str(f.x))
            if is_set(f.xbar):
                retval += 'xbar:\n %s \n'%(str(f.xbar))
        
        retval += '\nIndependent Function List:\n'
        retval += str([f.ID for f in self.independentFunctionList])
        
        retval += '\n\nDependent Function List:\n'
        retval += str([f.ID for f in self.dependentFunctionList])
        retval += '\n'
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
            f.args[0].x = x_list[nf]

        # traverse the computational tree
        for f in self.functionList:
            f.__class__.push_forward(f.func, f.args, Fout = f)


    def pullback(self, xbar_list):
        """
        Apply the pullback of the cotangent element, 
        
        e.g. for::
        
            y = y(x)
        
        compute::
        
            ybar dy(x) = xbar dx
            
        """
        
        if len(self.dependentFunctionList) == 0:
            raise Exception('You forgot to specify which variables are dependent!\n e.g. with cg.dependentFunctionList = [F1,F2]')

        # initial all xbar to zero
        for f in self.functionList:
            f.xbar_from_x()

        for nf,f in enumerate(self.dependentFunctionList):
            f.xbar[...] = xbar_list[nf]
            
        # print self
        for f in self.functionList[::-1]:
            # print 'pullback of f=',f.func.__name__
            f.__class__.pullback(f)
            # print self


    def plot(self, filename = None, method = None, orientation = 'TD'):
        """
        accepted filenames, e.g.:
        filename =
        'myfolder/mypic.png'
        'mypic.svg'
        etc.

        accepted methods
        method = 'dot'
        method = 'circo'

        accepted orientations:
        orientation = 'LR'
        orientation = 'TD'
        """
        
        try:
            import yapgvb
            
        except:
            print 'you will need yapgvb to plot graphs'
            return
            
        import os

        # checking filename and converting appropriately
        if filename == None:
            filename = 'computational_graph.png'

        if orientation != 'LR' and orientation != 'TD' :
            orientation = 'TD'

        if method != 'dot' and method != 'circo':
            method = 'dot'
        name, extension = filename.split('.')
        if extension != 'png' and extension != 'svg':
            print 'Only *.png or *.svg are supported formats!'
            print 'Using *.png now'
            extension = 'png'

        # print 'name=',name, 'extension=', extension

        # setting the style for the nodes
        g = yapgvb.Digraph('someplot')
        
        # add nodes
        for f in self.functionList:
            if f.func == Function.Id:
                g.add_node('%d'%f.ID, label = '%d %s'%(f.ID,f.func.__name__), shape = yapgvb.shapes.doublecircle,
                    color = yapgvb.colors.blue, fontsize = 10)
            else:
                g.add_node('%d'%f.ID, label = '%d %s'%(f.ID,f.func.__name__), shape = yapgvb.shapes.box,
                    color = yapgvb.colors.blue, fontsize = 10)
       
        # add edges
        nodes = list(g.nodes)
        for f in self.functionList:
            for a in numpy.ravel(f.args):
                if isinstance(a, Function):
                    nodes[a.ID] >> nodes[f.ID]
                
        # independent nodes
        for f in self.independentFunctionList:
            nodes[f.ID].shape = yapgvb.shapes.octagon
            
        # dependent nodes
        for f in self.dependentFunctionList:
            nodes[f.ID].shape = yapgvb.shapes.octagon
        
        g.layout(yapgvb.engines.circo)
        g.render(filename)

class Function(Ring):
    
    __array_priority__ = 2
    
    xbar = NotSet()
    setitem = NotSet()
    
    def __init__(self, x = None):
        """
        Creates a new function node that is a variable.
        """
        
        if type(x) != type(None):
            # create a Function node with value x referring to itself, i.e.
            # returning x when called
            cls = self.__class__
            cls.create(x, [self], cls.Id, self)
    
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
    def create(cls, x, fargs, func, f = None):
        """
        Creates a new function node.
        
        INPUTS:
            x           anything                            current value
            args        tuple of Function objects           arguments of the new Function node
            func        callable                            the function that can evaluate func(x)
            
        OPTIONAL:
            f           Function instance
            funcargs    tuple                               additional arguments to the function func
        
        """
        
        if not isinstance(fargs, list):
            raise ValueError('fargs must be of type list')
        
        if f == None:
            f = Function()
        f.x = x
        f.args = fargs
        f.func = func
        if cls.cgraph != None:
            f.ID = cls.get_ID()
            cls.cgraph.append(f)
        return f
        
    @classmethod
    def push_forward(cls, func, Fargs, Fout = None, setitem = None):
        """
        Computes the push forward of func
        
        INPUTS:
            func            callable            func(Fargs[0].x,...,Fargs[-1].x)
            Fargs           tuple               tuple of Function nodes
        """

        if not isinstance(Fargs,list):
            raise ValueError('Fargs has to be of type list')
        
        # STEP 1: extract arguments for func
        args = []
        for fa in Fargs:
            if isinstance(fa, cls):
                args.append(fa.x)
                
            else:
                args.append(fa)
        
        # STEP 2: call the function
        out  = func(*args)
        
        # STEP 3: create new Function instance for output
        if Fout == None:
            # this is called when push_forward is called by a function like mul,add, ...
            Fout = cls.create(out, Fargs, func)
            # return retval
        
        else:
            # this branch is called when Function(...) is used
            Fout.x = out
        
        # in case the function has side effects on a buffer
        # we need to store the values that are going to be changed
        if setitem != None:
            Fout.setitem = setitem
        
        return Fout
                     
             
            
            
    @classmethod
    def pullback(cls, F):
        """
        compute the pullback of the Function F
        
        e.g. if y = f(x)
        compute xbar as::
        
            ybar dy = ybar df(x) = ybar df/dx dx = xbar dx
            
        More specifically:
        
        (y1,y2) = f(x1,x2,const)
        
        where v is a constant argument.
        
        Examples:

            1) (Q,R) = qr(A)
            2) Q = getitem(qr(A),0)
        
        This function assumes that for each function f there is a corresponding function::
        
            pb_f(y1bar,y2bar,x1,x2,y1,y2,out=(x1bar, x2bar))
            
        The Function F contains information about its arguments, F.y and F.ybar.
        Thus, pullback(F) computes F.args[i].xbar
        """
        
        func_name = F.func.__name__
        
        # STEP 1: extract arguments
        args = []
        argsbar = []
        for a in F.args:
            if isinstance(a, cls):
                args.append(a.x)
                argsbar.append(a.xbar)
            else:
                args.append(a)
                argsbar.append(None)
        
        if isinstance(F.x,tuple):
            # case if the function F has several outputs, e.g. (y1,y2) = F(x)
            args = list(F.xbar) + args + list(F.x)
            f = eval('__import__("algopy.utpm").utpm.'+F.x[0].__class__.__name__+'.pb_'+func_name)            

        elif type(F.x) == type(None):
            # case if the function F has no output, e.g. None = F(x)
            f = eval('__import__("algopy.utpm").utpm.'+F.args[0].x.__class__.__name__+'.pb_'+func_name)
    
        elif numpy.isscalar(F.x):
            return lambda x: None
    
        else:
            # case if the function F has output, e.g. y1 = F(x)
            args = [F.xbar] + args + [F.x]
                
            # get the pullback function
            f = eval('__import__("algopy.utpm").utpm.'+F.x.__class__.__name__+'.pb_'+func_name)

        
        # STEP 2: call the pullback function
        kwargs = {'out': list(argsbar)}
        
        # print 'calling pullback function f=',f
        # print 'args = ',args
        # print 'kwargs = ',kwargs
        
        f(*args, **kwargs )
        
        # STEP 3: restore buffer values (i.e. if this is the pullback of the setitem function)
        
        if is_set(F.setitem):
            # print 'restoring value'
            # print 'F.setitem=', F.setitem
            F.args[0].x[F.setitem[0]] = F.setitem[1]
            
            # print 'F.args=',F.args
        
        return F
        
        
    @classmethod
    def totype(cls, x):
        """
        tries to convert x to an object of the class
        """
            
        if isinstance(x, cls):
            return x
        
        else:
            return cls(x)
    
    def xbar_from_x(self):
        if numpy.isscalar(self.x):
            self.xbar = 0.
            
        elif isinstance(self.x, tuple):
            self.xbar = tuple( [xi.zeros_like() for xi in self.x])
            
        elif self.x == None:
            pass
        else:
            if self.x.owndata == True or self.func == self.Id:
                self.xbar = self.x.zeros_like()
            else:
               
                # STEP 1: extract arguments for func
                args = []
                for fa in self.args:
                    if isinstance(fa, self.__class__):
                        args.append(fa.xbar)
                        # print fa.ID
                        
                    else:
                        args.append(fa)
                        
                # print self.func
                # # print 'self.x', self.x
                # print 'args', args
                # print self.func(*args)
                self.xbar = self.func(*args)
                # print self.xbar.shape
                # print tmp
                # self.xbar = self.func(
                # print len(self.args)
                # print self.args[0].x.strides
                # raise NotImplementedError('should implement that')
            
            
    def __getitem__(self, sl):
        return Function.push_forward(self.x.__class__.__getitem__,[self,sl])

    def __setitem__(self, sl, rhs):
        rhs = self.totype(rhs)
        store = self.x.__class__.__getitem__(self.x,sl).copy()
        # print 'storing ', store
        # print 'rhs = ',rhs
        return Function.push_forward(self.x.__class__.__setitem__,[self,sl,rhs], setitem = (sl,store))

    def __neg__(self):
        return self.__class__(-self.x)
    
    # FIXME: implement the inplace operations for better efficiency    
    # def __iadd__(self,rhs):
        # rhs = self.totype(rhs)
        # return Function.push_forward(self.x.__class__.__iadd__,[self,rhs])
        
        
    def __add__(self,rhs):
        rhs = self.totype(rhs)
        return Function.push_forward(self.x.__class__.__add__,[self,rhs])

    def __sub__(self,rhs):
        rhs = self.totype(rhs)
        return Function.push_forward(self.x.__class__.__sub__,[self,rhs])

    def __mul__(self,rhs):
        rhs = self.totype(rhs)
        return Function.push_forward(self.x.__class__.__mul__,[self,rhs])

    def __div__(self,rhs):
        rhs = self.totype(rhs)
        return Function.push_forward(self.x.__class__.__div__,[self,rhs])
        
    def __radd__(self,lhs):
        return self + lhs

    def __rsub__(self,lhs):
        return -self + lhs

    def __rmul__(self,lhs):
        return self * lhs

    def __rdiv__(self, lhs):
        lhs = self.__class__.totype(lhs)
        return lhs/self
    
    @classmethod
    def dot(cls, lhs,rhs):
        lhs = cls.totype(lhs)
        rhs = cls.totype(rhs)
        
        try:
            out = Function.push_forward(lhs.x.__class__.dot, [lhs,rhs])
            return out
        except:
            out = Function.push_forward(rhs.x.__class__.dot, [lhs,rhs])
            return out

    def log(self):
         return Function.push_forward(self.x.__class__.log, [self])

    def exp(self):
         return Function.push_forward(self.x.__class__.exp, [self])

    def inv(self):
         return Function.push_forward(self.x.__class__.inv, [self])
         
    def qr(self):
         return Function.push_forward(self.x.__class__.qr, [self])

    def eigh(self):
         return Function.push_forward(self.x.__class__.eigh, [self])

    def solve(self,rhs):
        return Function.push_forward(self.x.__class__.solve, [self,rhs])
        
    def trace(self):
        return Function.push_forward(self.x.__class__.trace, [self])
        
    def transpose(self):
        return Function.push_forward(self.x.__class__.transpose, [self])
        
    T = property(transpose)
    
    def get_shape(self):
        return self.x.shape
    shape = property(get_shape)
 
