import numpy as np
import uncertainties

class EVAL:
    def __init__(self, f, x):
        self.f = f
        self.x = x
        
    def gradient(self, x):
        N = len(x)
        sx = np.array([uncertainties.ufloat((x[i],np.inf)) for i in range(N)])
        sf = self.f(sx)
        return np.array([sf.derivatives[sx[i]] for i in range(N)])










