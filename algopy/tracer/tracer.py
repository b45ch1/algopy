import numpy
import algopy
import operator
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
    
    """

    def __init__(self):
        self.functionCount = 0
        self.functionList = []
        self.dependentFunctionList = []
        self.independentFunctionList = []
        Function.cgraph = self
        
    def trace_on(self):
        Function.cgraph = self
        return self
    
    def trace_off(self):
        Function.cgraph = None
        return self
        
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
            retval += 'class:    %s \n'%( str(f.x.__class__))

            retval += 'x:\n    %s \n'%( str(f.x))
            if is_set(f.xbar):
                retval += 'xbar:\n %s \n'%(str(f.xbar))
        
        retval += '\nIndependent Function List:\n'
        retval += str([f.ID for f in self.independentFunctionList])
        
        retval += '\n\nDependent Function List:\n'
        retval += str([f.ID for f in self.dependentFunctionList])
        retval += '\n'
        return retval
        
    def pushforward(self,x_list):
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
            f.__class__.pushforward(f.func, f.args, Fout = f)


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
            # print 'f=',f.func.__name__
            f.xbar_from_x()
            
        # print 'before pullback',self

        for nf,f in enumerate(self.dependentFunctionList):
            f.xbar[...] = xbar_list[nf]
            
        # print self
        for f in self.functionList[::-1]:
            # print 'pullback of f=',f.func.__name__
            f.__class__.pullback(f)
            # print self
            
    def function(self, x_list):
        """ computes the function of a function y = f(x_list), where y is a scalar 
        and x_list is a list or tuple of input arguments.
        
        The computation is performed using the stored computational graph.
        Using this function one can check if the CGraph is correct by comparing the value
        to the normal function evaluation in Python.
        """

        self.pushforward(x_list)
        return [x.x for x in self.dependentFunctionList]
            
            

    def gradient(self, x_list):
        """ computes the gradient of a function y = f(x_list), where y is a scalar 
        and x_list is a list or tuple of input arguments.
        The computation is performed in thein the reverse mode of AD.
        x_list contains the independent variables at which the gradient should be evaluated.
        
        Warning: this function does _not_ compute the Jacobian.
        """
        
        if self.dependentFunctionList[0].ndim != 0:
            raise Exception('you are trying to compute the gradient of a non-scalar valued function')
        
        utpm_x_list = [algopy.UTPM(numpy.asarray(x).reshape((1,1) + numpy.shape(x))) for x in x_list]
        
        # print utpm_x_list
        self.pushforward(utpm_x_list)
        
        # print self
        ybar =  self.dependentFunctionList[0].x.zeros_like()
        ybar.data[0,:] = 1.
        self.pullback([ybar])
        
        return [x.xbar.data[0,0] for x in self.independentFunctionList]
        # print self
        
    def jacobian(self, x_list):
        """ computes the Jacobian of a function F:R^N --> R^M
        """
        M = self.dependentFunctionList[0].size
        
        utpm_x_list = []
        for n,x in enumerate(x_list):
            tmp = numpy.zeros((1,M) + numpy.shape(x))
            tmp[0,...] = x
            utpm_x_list.append(algopy.UTPM(tmp))

        self.pushforward(utpm_x_list)
        
        ybar =  algopy.UTPM(numpy.zeros((1,M,M)))
        ybar.data[0,:,:] = numpy.eye(M)
        self.pullback([ybar])
        
        return [x.xbar.data[0,:] for x in self.independentFunctionList]
        # print self        
        
    def hessian(self, x_list):
        """ computes the Hessian
        
        H = self.hessian(x_list)
        
        Parameters
        ----------
        x_list: list of array_like
            inputs as in self.independentFunctionList
            
        Returns
        -------
        H: array
            two-dimensional array containing the Hessian
            
        """
        if len(x_list) != 1:
            raise NotImplementedError('')
        
        tmp = [numpy.ravel(x) for x in x_list]
        tmp = numpy.concatenate(tmp)
        tmp = algopy.UTPM.init_jacobian(tmp)
        
        utpm_x_list = []
        a = 0
        for x in x_list:
            b = numpy.prod(x.shape)
            utpm_x_list.append(numpy.reshape(tmp[a:b],x.shape))

        self.pushforward(utpm_x_list)
        
        ybar =  self.dependentFunctionList[0].x.zeros_like()
        ybar.data[0,:] = 1.
        self.pullback([ybar])
        
        return numpy.array([x.xbar.data[1,:] for x in self.independentFunctionList])
        
    def hess_vec(self, x_list, v_list):
        """ computes the Hessian vector product  dot(H,v)
        
        Hv = self.hess_vec(x_list, v_list)
        
        Parameters
        ----------
        x_list: list of array_like
            inputs as in self.independentFunctionList
            
        v_list: list of array_like
            input directions
            
        Returns
        -------
        Hv: array
            one-dimensional array containing the Hessian vector product
            
        """
        if len(x_list) != 1 or len(v_list) != 1:
            raise NotImplementedError('')        
            
        x = x_list[0]
        v = v_list[0]
        xtmp = numpy.zeros((2,1) + x.shape)
        xtmp[0,0] = x; xtmp[1,0] = v
        xtmp = algopy.UTPM(xtmp)
        
        self.pushforward([xtmp])
        ybar =  self.dependentFunctionList[0].x.zeros_like()
        ybar.data[0,:] = 1.
        self.pullback([ybar])
        
        return self.independentFunctionList[0].xbar.data[1,0]
        
    def vec_hess(self, w, x_list):
        """ computes  the hessian of dot(w, F(x)), where F:R^N ---> R^M
        
        Hv = self.vec_hess(w, x_list)
        
        Parameters
        ----------
        w: array_like
            
        x_list: list of array_like
            inputs as in self.independentFunctionList
            
        v_list: list of array_like
            input directions
            
        Returns
        -------
        Hv: array
            one-dimensional array containing the Hessian vector product
            
        """
        if len(x_list) != 1:
            raise NotImplementedError('')        
            
        tmp = [numpy.ravel(x) for x in x_list]
        tmp = numpy.concatenate(tmp)
        tmp = algopy.UTPM.init_jacobian(tmp)
        
        utpm_x_list = []
        a = 0
        for x in x_list:
            b = numpy.prod(x.shape)
            utpm_x_list.append(numpy.reshape(tmp[a:b],x.shape))

        self.pushforward(utpm_x_list)
        
        ybar = self.dependentFunctionList[0].x.zeros_like()
        ybar.data[0,:] = w
        self.pullback([ybar])
        
        return numpy.array([x.xbar.data[1,:] for x in self.independentFunctionList])      
        
    def vec_hess_mat(self, lagra, x_list, V_list):
        """ computes  d^2(lagra F) V, where  F:R^N ---> R^M
        
        lHv = self.vec_hess_mat(w, x, V)
        
        Parameters
        ----------
        lagra: array_like
            "Lagrange multipliers"
            
        x_list: list of array_like
            inputs as in self.independentFunctionList
            
        V_list: list of array_like
            input directions
            
        Returns
        -------
        lHv: array
            two-dimensional array containing the result
            
        """
        if len(x_list) != 1 or len(V_list) != 1:
            raise NotImplementedError('')
            
        raise NotImplementedError('this function does not work correctly yet')
        
        x = x_list[0]
        V = V_list[0]
        
        P = V.shape[1]
        
        xtmp = numpy.zeros((2,P) + x.shape)
        xtmp[0,:] = x; xtmp[1,...] = V
        xtmp = algopy.UTPM(xtmp)
        
        self.pushforward([xtmp])
        ybar =  self.dependentFunctionList[0].x.zeros_like()
        print 'ybar.shape=',ybar.shape
        ybar.data[0,...] = lagra
        self.pullback([ybar])
        
        return self.independentFunctionList[0].xbar.data[1,:]            



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
        method = 'fdp'
        method = 'twopi'
        method = 'neato'

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
        
        method_list = ['dot','circo','fdp','twopi','neato']

        # checking filename and converting appropriately
        if filename == None:
            filename = 'computational_graph.png'

        if orientation != 'LR' and orientation != 'TD' :
            orientation = 'TD'

        if not any([method == m for m in method_list]):
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
        
        g.layout(method)
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
        retstr = 'Function:\n'
        retstr += 'ID = %s\n'%str(self.get_ID())
        retstr += 'function = %s\n'%str(self.func)
        return retstr

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
    def pushforward(cls, func, Fargs, Fout = None, setitem = None):
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
        # print 'func=',func
        # print 'args=',args
        out  = func(*args)
        
        # STEP 3: create new Function instance for output
        if Fout == None:
            # this is called when pushforward is called by a function like mul,add, ...
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
    
        elif numpy.isscalar(F.x) or isinstance(F.x, numpy.ndarray):
            return lambda x: None
    
        elif isinstance(F.x, algopy.UTPM):
            
            # case if the function F has output, e.g. y1 = F(x)
            args = [F.xbar] + args + [F.x]
            
            # print '-------'
            # print func_name
            # print F.x
            # print F.xbar
            
            # get the pullback function
            f = eval('__import__("algopy.utpm").utpm.'+F.x.__class__.__name__+'.pb_'+func_name)

        elif func_name == '__getitem__' or func_name == 'getitem':
            return  lambda x: None
            # raise NotImplementedError('should implement that')
            
        # elif func_name == '__getitem__':
        #     return  lambda x: None
            # raise NotImplementedError('should implement that')
            
        # STEP 2: call the pullback function
        kwargs = {'out': list(argsbar)}
        
        # print 'func_name = ',func_name
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
        """
        
        Warning for the faint-hearted: this function is quite a hack and should be refactored
        
        each Function instance has the attribute x.
        xbar_from_x creates a corresponding attribute xbar that is a copy of x but filled with zeros.
        
        it also supports functions with multiple outputs of the form
        
        (y1, y2) = f(x1,x2,x3)
        
        i.e. f.x = (y1, y2) hence the corresponding bar value is
        f.xbar = (y1bar, y2bar)
        
        
        In the case that e.g. y2 is not an UTPM instance, this function generates
        f.xbar = (y1bar, None)
        
        """
        if numpy.isscalar(self.x):
            self.xbar = 0.
        
        
        # case that the function f had a tuple of outputs
        elif isinstance(self.x, tuple):
            tmp = []
            
            for xi in self.x:
                if isinstance(xi, algopy.UTPM):
                    tmp.append(xi.zeros_like())
                else:
                    tmp.append(None)
            self.xbar = tuple(tmp)
            # self.xbar = tuple( [xi.zeros_like() for xi in self.x])
            
        elif self.x == None:
            pass
        
        # case that the output of the function is an UTPM instance
        elif isinstance(self.x, algopy.UTPM):

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
          
        else:
            import copy
            
            # this is a hack to get L,Q,b = eigh1 to work, should be fixed soon!
            self.xbar = None # copy.deepcopy(self.x)
            
    def __getitem__(self, sl):
        return Function.pushforward(operator.getitem ,[self,sl])

    def __setitem__(self, sl, rhs):
        rhs = self.totype(rhs)
        store = operator.getitem(self.x,sl).copy()
        # print 'storing ', store
        # print 'rhs = ',rhs
        return Function.pushforward(operator.setitem,[self,sl,rhs], setitem = (sl,store))

    def __neg__(self):
        return Function.pushforward(self.x.__class__.__neg__,[self])
    
    # FIXME: implement the inplace operations for better efficiency    
    # def __iadd__(self,rhs):
        # rhs = self.totype(rhs)
        # return Function.pushforward(self.x.__class__.__iadd__,[self,rhs])
        
        
    def __add__(self,rhs):
        rhs = self.totype(rhs)
        return Function.pushforward(operator.add,[self,rhs])

    def __sub__(self,rhs):
        rhs = self.totype(rhs)
        return Function.pushforward(operator.sub,[self,rhs])

    def __mul__(self,rhs):
        rhs = self.totype(rhs)
        return Function.pushforward(operator.mul,[self,rhs])

    def __div__(self,rhs):
        rhs = self.totype(rhs)
        return Function.pushforward(operator.div,[self,rhs])
        
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
        
        out = Function.pushforward(algopy.dot, [lhs,rhs])
        return out

    def log(self):
         return Function.pushforward(algopy.log, [self])

    def exp(self):
         return Function.pushforward(algopy.exp, [self])
         
    def sin(self):
         return Function.pushforward(algopy.sin, [self])
         
    def tan(self):
         return Function.pushforward(algopy.tan, [self])
         
    def cos(self):
         return Function.pushforward(algopy.cos, [self])
         
    def sqrt(self):
         return Function.pushforward(algopy.sqrt, [self])
    
    def __pow__(self, r):
         return Function.pushforward(operator.pow, [self, r])
         
    def sum(self, axis=None, dtype=None, out=None):
         return Function.pushforward(numpy.sum, [self, axis, dtype, out])
         
    def inv(self):
         return Function.pushforward(algopy.inv, [self])
         
    def qr(self):
         return Function.pushforward(algopy.qr, [self])
         
    def qr_full(self):
         return Function.pushforward(algopy.qr_full, [self])
         
    def diag(self):
         return Function.pushforward(algopy.diag, [self])

    def eigh(self):
         return Function.pushforward(algopy.eigh, [self])
         
    def eigh1(self):
         return Function.pushforward(algopy.eigh1, [self])

    def solve(self,rhs):
        return Function.pushforward(algopy.solve, [self,rhs])
        
    def trace(self):
        return Function.pushforward(algopy.trace, [self])
        
    def transpose(self):
        return Function.pushforward(algopy.transpose, [self])
        
    def tril(self):
        return Function.pushforward(algopy.tril, [self])

    def triu(self):
        return Function.pushforward(algopy.triu, [self])        
                
    def symvec(self, UPLO='F'):
        return Function.pushforward(algopy.symvec, [self, UPLO])
        
    def vecsym(self):
        return Function.pushforward(algopy.vecsym, [self])
        
    T = property(transpose)
    
    def get_shape(self):
        return self.x.shape
    shape = property(get_shape)
    
    def get_ndim(self):
        return numpy.ndim(self.x)
    ndim = property(get_ndim)
    
    def get_size(self):
        return self.x.size
    size = property(get_size)    
 
