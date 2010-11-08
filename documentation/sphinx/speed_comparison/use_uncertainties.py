import numpy as np
import uncertainties

class EVAL:
    def __init__(self, f, x):
        self.f = f
        self.x = x
        N = len(x)
        self.sx = np.array([uncertainties.ufloat((x[i],np.inf)) for i in range(N)])
        self.sf = f(self.sx)
        
    def gradient(self, x):
        N = len(x)
        return np.array([self.sf.derivatives[self.sx[i]] for i in range(N)])










