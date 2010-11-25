import algopy
import numpy


class Model:
    def eval_f(self, x):
        return (x[0]-10)**2 + 5*(x[1]-12)**2 + x[2]**4 + 3*(x[3]-11)**2 + 10*x[4]**6
        + 7*x[5]**2 + x[6]**4 - 4*x[5]*x[6] - 10*x[5] - 8*x[6]
  
    def eval_g(self, x):
        out = algopy.zeros(3, dtype=x)
        out[0] = 2*x[0]**2 + 3*x[1]**4 + x[2] + 4*x[3]**2 + 5*x[4]
        out[1] = 7*x[0] + 3*x[1] + 10*x[2]**2 + x[3] - x[4]
        out[2] = 23*x[0] + x[1]**2 + 6*x[5]**2 - 8*x[6]  -4*x[0]**2 - x[1]**2 + 3*x[0]*x[1] -2*x[2]**2 - 5*x[5]+11*x[6]
        return out
    
    def eval_Lagrangian(self, lam,x):
        return self.eval_f(x) #+ algopy.dot(lam, self.eval_g(x))
        
    def eval_grad_f_forward(self, x):
        x = algopy.UTPM.init_jacobian(x)
        return algopy.UTPM.extract_jacobian(self.eval_f(x))
        
    def eval_jac_g_forward(self, x):
        x = algopy.UTPM.init_jacobian(x)
        return algopy.UTPM.extract_jacobian(self.eval_g(x))
        
    def eval_jac_vec_g_forward(self, x, v):
        x = algopy.UTPM.init_jac_vec(x, v)
        return algopy.UTPM.extract_jac_vec(self.eval_g(x))
        
    def eval_grad_Lagrangian_forward(self, lam, x):
        return self.eval_grad_f_forward(x) + algopy.dot(lam, self.eval_jac_g_forward(x))
        
    def eval_hess_Lagrangian_forward(self, lam, x):
        x = algopy.UTPM.init_hessian(x)
        return algopy.UTPM.extract_hessian(6, self.eval_Lagrangian(lam, x))
    
    def trace_eval_f(self, x):
        cg = algopy.CGraph()
        x = algopy.Function(x)
        y = self.eval_f(x)
        cg.trace_off()
        cg.independentFunctionList = [x]
        cg.dependentFunctionList = [y]
        self.cg = cg
        
    def eval_grad_f_reverse(self, x):
        return self.cg.gradient([x])
        
        

lam = numpy.array([1,1,1],dtype=float)
x = numpy.array([1,2,3,4,0,1,1],dtype=float)

m = Model()
print m.eval_grad_f_forward(x)
print m.eval_jac_g_forward(x)
print m.eval_jac_vec_g_forward(x,[1,0,0,0,0,0,0])
print m.eval_grad_Lagrangian_forward(lam, x)

# print m.eval_hess_Lagrangian_forward(lam, x)

m.trace_eval_f(x)
print m.eval_grad_f_reverse(x)








