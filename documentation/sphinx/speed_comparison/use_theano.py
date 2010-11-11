import theano 
import numpy
import time


import numpy
import adolc

class EVAL:
    def __init__(self, f, x):
        self.f = f
        self.x = x
        self.A = f.A

        x = theano.tensor.dvector('x')
        A = theano.tensor.dmatrix('A')
        y = 0.5*theano.dot(x*x,theano.dot(A,x))
        gy = theano.tensor.grad(y,x)
        hy = theano.tensor.grad(gy,x)
        self.eval_grad = theano.function([x,A], gy)
        self.eval_hess = theano.function([x,A], hy)
        
        
    # def function(self, x):
    #     return adolc.function(0,x)
        
    def gradient(self, x):
        return self.eval_grad(x,self.A)

    def hessian(self, x):
        return self.eval_hess(x,self.A)








# #A = numpy.random.rand(2,2)
# N = 100

# x = theano.tensor.dvector('x')
# A = theano.tensor.dmatrix('A')

# y = 0.5 * theano.dot(x, theano.dot(A,x))

# gy = theano.tensor.grad(y,x)
# dlogistic = theano.function([x,A], gy)

# A = numpy.random.rand(N,N)
# A = numpy.dot(A.T,A)
# x = numpy.ones(N)
# start_time = time.time()
# g = dlogistic(x,A)
# end_time = time.time()


# print end_time - start_time

# print g - numpy.dot(A,x) 
