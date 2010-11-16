import numpy
import adolc

class EVAL:
    def __init__(self, f, x, test = 'f'):
        self.f = f
        self.x = x

        adolc.trace_on(0)
        ax = adolc.adouble(x)
        adolc.independent(ax)
        y = f(ax)
        adolc.dependent(y)
        adolc.trace_off()
        
    def function(self, x):
        return adolc.function(0,x)
        
    def gradient(self, x):
        return adolc.gradient(0,x)

    def hessian(self, x):
        return adolc.hessian(0,x)







