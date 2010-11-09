import numpy as np
import FuncDesigner

class EVAL:
    def __init__(self, f, x):
        self.f = f
        self.x = x.copy()

        sx = FuncDesigner.oovar('x')
        sy = f(sx)[0,0]
        
        self.sx = sx
        self.sy = sy
        
    def function(self, x):
        point = {self.sx:x}
        return self.sy(point)
        
    def gradient(self, x):
        point = {self.sx:x}
        # print point
        # print self.sy
        return self.sy.D(point)[self.sx]
        
    # def hessian(self, x):
    #     return adolc.hessian(0,x)
     
