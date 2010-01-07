"""
The tracer records the sequence of elementary operations (+,-,*,/,sin,cos,dot,solve,...).
The sequence of elementary operations is necessary to implement the
reverse mode of AD.

The tracer defined in the file works in the following way:
    
    * each variable that should be traced is an object of the class Function
    * the Function object is operations aware, i.e. it knows that is used in a
    function.
    * This operations awareness is accomplished by operator overloading, i.e.
    when Function(1)*Function(2) is called, it is equivalent to
    Function(1).__mul__(Function(2))
    and the __mul__ operator is defined to record the operation
    * The sequence of recorded operations is stored in a DAG (computational graph)
    * The DAG has the special property that each node in the graph knows its
    parents. This is necessary to revert the control flow direction.
"""

import numpy
import copy


#def combine_function_blocks(in_X):
    #"""
    #FZ = combine_blocks([[FX1,FX2],[FX3,FX4]])

    #combine_blocks returns a Function object where the input matrix of functions
    #is combined in a function of the combined matrix

    #"""
    #import algopy.utp.utpm

    #X = numpy.array(in_X)
    #AX = numpy.reshape(map(lambda v: getattr(v,'x'), X.ravel()), numpy.shape(X))
    #AY = algopy.utp.utpm.combine_blocks(AX)
    #return Function(AY)

class Function:
    """
    Function node of the Computational Graph CGraph defined below.
    It is only a container class with the true value saved in self.x
    Its parents are saved as list in self.args.

    Before instances of the Function class can be made, one has to register a
    computational graph. This is simply done by calling creating an instance of
    a computational graph:
    >>>CGraph()

    The CGraph init function registers this instance in the Function class:
    CGraph.__init__(self):
        Function.cgraph = self
    """
    
    
    
    def __init__(self, args, function_type='var'):

        # register all supported elementary functions here
        if function_type == 'var':
            if type(args) == list:
                self.type = 'com'
            else:
                self.type = 'var'
        elif function_type == 'id':
            self.type = 'id'
        elif function_type == 'const':
            self.type = 'const'
        elif function_type == 'com':
            self.type = 'com'
        elif function_type == 'neg':
            self.type = 'neg'
        elif function_type == 'add':
            self.type = 'add'
        elif function_type == 'sub':
            self.type = 'sub'
        elif function_type == 'mul':
            self.type = 'mul'
        elif function_type == 'div':
            self.type = 'div'
        elif function_type == 'dot':
            self.type = 'dot'
        elif function_type == 'trace':
            self.type = 'trace'
        elif function_type == 'inv':
            self.type = 'inv'
        elif function_type == 'JT':
            self.type = 'JT'            
        elif function_type == 'solve':
            self.type = 'solve'
        elif function_type == 'trans':
            self.type = 'trans'
        else:
            raise NotImplementedError('function_type "%s" is unknown, please add to Function.__init__'%function_type)

        # save the arguments and compute the value of the operation and save it in self.x
        self.args = args
        self.x = self.eval()
        #self.xbar_from_x()

        # update the computational graph
        self.id = self.cgraph.functionCount
        self.cgraph.functionCount += 1
        self.cgraph.functionList.append(self)


    # convenience functions
    # ---------------------
    def as_function(self, in_x):
        if not isinstance(in_x, Function):
            (D,P,N,M) = numpy.shape(self.x.TC)
            fun = Mtc(numpy.zeros([D,P]+ list(numpy.shape(in_x))))
            fun = Function(fun, function_type='const')
            for p in range(P):
                fun.x.TC[0,p,:,:] = in_x
            return fun
        return in_x

    def xbar_from_x(self):
        if numpy.isscalar(self.x):
            self.xbar = 0.
        else:
            self.xbar = self.x.zeros_like()


    def __str__(self):
        try:
            ret = '%s%s:\n(x=\n%s)\n(xbar=\n%s)'%(self.type,str(self.id),str(self.x),str(self.xbar))
        except:
            ret = '%s%s:(x=%s)'%(self.type,str(self.id),str(self.x))
        return ret

    def __repr__(self):
        return self.__str__()
    # ---------------------


    # overloaded elementary operations
    # --------------------------------
    def __neg__(self):
        return Function([self], function_type='neg')
    
    def __add__(self,rhs):
        rhs = self.as_function(rhs)
        return Function([self, rhs], function_type='add')

    def __sub__(self,rhs):
        rhs = self.as_function(rhs)
        return Function([self, rhs], function_type='sub')

    def __mul__(self,rhs):
        rhs = self.as_function(rhs)
        return Function([self, rhs], function_type='mul')

    def __div__(self,rhs):
        rhs = self.as_function(rhs)
        return Function([self, rhs], function_type='div')

    def __radd__(self,lhs):
        return self + lhs

    def __rsub__(self,lhs):
        return -self + lhs

    def __rmul__(self,lhs):
        return self * lhs

    def __rdiv__(self, lhs):
        lhs = Function(Tc(lhs), function_type='const')
        return lhs/self

    def dot(self,rhs):
        rhs = self.as_function(rhs)
        return Function([self, rhs], function_type='dot')

    def trace(self):
        return Function([self], function_type='trace')

    def inv(self):
        return Function([self], function_type='inv')
        
    def toTransposedJacobian(self):
        return Function([self], function_type='JT')

    def get_shape(self):
        return numpy.shape(self.x)

    shape = property(get_shape)

    def solve(self,rhs):
        rhs = self.as_function(rhs)
        return Function([self, rhs], function_type='solve')

    def transpose(self):
        return Function([self], function_type='trans')

    def get_transpose(self):
        return self.transpose()
    def set_transpose(self,x):
        raise NotImplementedError('???')
    T = property(get_transpose, set_transpose)


    def eval(self):
        """
        one step in the forward evaluation
        """
        if self.type == 'var':
            return self.args

        elif self.type == 'const':
            return self.args

        elif self.type == 'com':
            import algopy.utp.utpm
            X = numpy.array(self.args)
            AX = numpy.reshape(map(lambda v: getattr(v,'x'), X.ravel()), numpy.shape(X))
            AY = algopy.utp.utpm.combine_blocks(AX)
            return AY

        elif self.type == 'neg':
            return -self.args[0].x

        elif self.type == 'add':
            return self.args[0].x + self.args[1].x

        elif self.type == 'sub':
            return self.args[0].x - self.args[1].x

        elif self.type == 'mul':
            return self.args[0].x * self.args[1].x

        elif self.type == 'div':
            return self.args[0].x.__div__(self.args[1].x)

        elif self.type == 'dot':
            return self.args[0].x.dot(self.args[1].x)

        elif self.type == 'trace':
            return self.args[0].x.trace()

        elif self.type == 'inv':
            return self.args[0].x.inv()
            
        elif self.type == 'JT':
            return self.args[0].x.JT()
            
        elif self.type == 'solve':
            return self.args[0].x.solve(self.args[1].x)

        elif self.type == 'trans':
            return self.args[0].x.transpose()

        else:
            raise Exception('Unknown function "%s". Please add rule to Mtc.eval()'%self.type)

    def reval(self):
        """
        one step in the reverse evaluation
        """
        if self.type == 'var':
            pass

        elif self.type == 'add':
            self.args[0].xbar += self.xbar
            self.args[1].xbar += self.xbar

        elif self.type == 'sub':
            self.args[0].xbar += self.xbar
            self.args[1].xbar -= self.xbar

        elif self.type == 'mul':
            self.args[0].xbar += self.xbar * self.args[1].x
            self.args[1].xbar += self.xbar * self.args[0].x

        elif self.type == 'div':
            self.args[0].xbar += self.xbar.__div__(self.args[1].x)
            self.args[1].xbar += self.xbar * self.args[0].x.__div__(self.args[1].x * self.args[1].x)

        elif self.type == 'dot':
            self.args[0].xbar +=  self.xbar.dot(self.args[1].x.T)
            self.args[1].xbar +=  self.args[0].x.T.dot(self.xbar)

        elif self.type == 'trace':
            N = self.args[0].xbar.shape[0]
            for n in range(N):
                self.args[0].xbar[n,n] += self.xbar

        elif self.type == 'inv':
            self.args[0].xbar -= self.x.T.dot(self.xbar.dot(self.x.T))
            
        # elif self.type == 'JT':
            # self.args[0].xbar += self.x.T.dot(self.xbar.dot(self.x.T))            

        elif self.type == 'solve':
            raise NotImplementedError

        elif self.type == 'trans':
            self.args[0].xbar += self.xbar.transpose()

        elif self.type == 'com':
            Rb,Cb = shape(self.args)
            #print 'xbar.shape()=',self.xbar.shape()
            args = asarray(self.args)
            rows = []
            cols = []
            #print type(args)
            for r in range(Rb):
                rows.append(args[r,0].shape[0])
            for c in range(Cb):
                cols.append(args[0,c].shape[1])

            #print rows
            #print cols

            rowsums = [ int(sum(rows[:r])) for r in range(0,Rb+1)]
            colsums = [ int(sum(cols[:c])) for c in range(0,Cb+1)]

            #print rowsums
            #print colsums
            #print 'shape of xbar=', shape(self.xbar.TC)
            #print 'shape of x=', shape(self.x.TC)

            for r in range(Rb):
                for c in range(Cb):
                    #print 'args[r,c].xbar=\n',args[r,c].xbar.shape()
                    #print 'rhs=\n', self.xbar[rowsums[r]:rowsums[r+1],colsums[c]:colsums[c+1]].shape()

                    args[r,c].xbar.TC[:,:,:,:] += self.xbar.TC[:,:,rowsums[r]:rowsums[r+1],colsums[c]:colsums[c+1]]

        else:
            raise Exception('Unknown function "%s". Please add rule to Mtc.reval()'%self.type)


class CGraph:
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

    def __str__(self):
        return 'vertices:\n' + str(self.functionList)
        
    def node_shapes(self):
        retval = ''
        for nf, f in enumerate(self.functionList):
            retval += 'Function %s%d:  '%(f.type,nf) + str(f.shape) + '\n'
        return retval

    def forward(self,x):
        # populate independent arguments with new values
        for nf,f in enumerate(self.independentFunctionList):
            f.args = x[nf]

        # traverse the computational tree
        for f in self.functionList:
            f.x = f.eval()

    def reverse(self,xbar):
        if numpy.size(self.dependentFunctionList) == 0:
            print 'You forgot to specify which variables are dependent!\n e.g. with cg.dependentFunctionList = [F1,F2]'
            return

        # initial all xbar to zero
        for f in self.functionList:
            f.xbar_from_x()

        for nf,f in enumerate(self.dependentFunctionList):
            f.xbar = xbar[nf]

        for f in self.functionList[::-1]:
            f.reval()

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

        import pygraphviz
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

        print 'name=',name, 'extension=', extension

        # setting the style for the nodes
        A = pygraphviz.agraph.AGraph(directed=True, strict = False, rankdir = orientation)
        A.node_attr['fillcolor']= '#ffffff'
        A.node_attr['shape']='rect'
        A.node_attr['width']='0.5'
        A.node_attr['height']='0.5'
        A.node_attr['fontcolor']="#000000"
        A.node_attr['style']='filled'
        A.node_attr['fixedsize']='true'

        # build graph

        for f in self.functionList:
            if f.type == 'var' or f.type == 'const':
                A.add_node(f.id)
                continue
            for a in numpy.ravel(f.args):
                A.add_edge(a.id, f.id)
                #e = A.get_edge(a.source.id, f.id)
                #e.attr['color']='green'
                #e.attr['label']='a'

        # extra formatting for the dependent variables
        for f in self.dependentFunctionList:
            s = A.get_node(f.id)
            s.attr['fillcolor'] = "#FFFFFF"
            s.attr['fontcolor']='#000000'

        # applying the style for the nodes
        for nf,f in enumerate(self.functionList):
            s = A.get_node(nf)
            vtype = f.type

            if vtype == 'add':
                s.attr['label']='+%d'%nf

            elif vtype == 'sub':
                s.attr['label']='-%d'%nf

            elif vtype == 'mul':
                s.attr['label']='*%d'%nf

            elif vtype == 'div':
                s.attr['label']='/%d'%nf

            elif vtype == 'var':
                s.attr['fillcolor']="#FFFFFF"
                s.attr['shape']='circle'
                s.attr['label']= 'v_%d'%nf
                s.attr['fontcolor']='#000000'
            elif vtype == 'const':
                s.attr['fillcolor']="#AAAAAA"
                s.attr['shape']='triangle'
                s.attr['label']= 'c_%d'%nf
                s.attr['fontcolor']='#000000'

            elif vtype == 'dot':
                s.attr['label']='dot%d'%nf

            elif vtype == 'com':
                s.attr['label']='com%d'%nf

            elif vtype == 'trace':
                s.attr['label']='tr%d'%nf

            elif vtype == 'inv':
                s.attr['label']='inv%d'%nf

            elif vtype == 'solve':
                s.attr['label']='slv%d'%nf

            elif vtype == 'trans':
                s.attr['label']='T%d'%nf
        #print A.string() # print to screen


        A.write('%s.dot'%name)
        os.system('%s  %s.dot -T%s -o %s.%s'%(method, name, extension, name, extension))