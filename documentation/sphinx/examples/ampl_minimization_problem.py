
import algopy
import numpy


class Model:

    # Function evaluations
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
        return self.eval_f(x) + algopy.dot(lam, self.eval_g(x))


    # Forward Mode Derivative Evaluations
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
        return algopy.UTPM.extract_hessian(x.size, self.eval_Lagrangian(lam, x))

    def eval_vec_hess_g_forward(self, w, x):
        x = algopy.UTPM.init_hessian(x)
        tmp = algopy.dot(w, self.eval_g(x))
        return algopy.UTPM.extract_hessian(x.size, tmp)

    # Reverse Mode Derivative Evaluations
    def trace_eval_f(self, x):
        cg = algopy.CGraph()
        x = algopy.Function(x)
        y = self.eval_f(x)
        cg.trace_off()
        cg.independentFunctionList = [x]
        cg.dependentFunctionList = [y]
        self.cg = cg

    def trace_eval_g(self, x):
        cg2 = algopy.CGraph()
        x = algopy.Function(x)
        y = self.eval_g(x)
        cg2.trace_off()
        cg2.independentFunctionList = [x]
        cg2.dependentFunctionList = [y]
        self.cg2 = cg2

    def eval_grad_f_reverse(self, x):
        return self.cg.gradient(x)

    def eval_jac_g_reverse(self, x):
        return self.cg2.jacobian(x)

    def eval_hess_f_reverse(self, x):
        return self.cg.hessian(x)

    def eval_hess_vec_f_reverse(self, x, v):
        return self.cg.hess_vec(x,v)

    def eval_vec_hess_g_reverse(self, w, x):
        return self.cg2.vec_hess(w, x)

    def eval_grad_Lagrangian_reverse(self, lam, x):
        return self.cg.gradient(x) + self.cg2.vec_jac(lam, x)

    def eval_hess_Lagrangian_reverse(self, lam, x):
        return self.cg.hessian(x) + self.cg2.vec_hess(lam, x)


lam = numpy.array([1,1,1],dtype=float)
x = numpy.array([1,2,3,4,0,1,1],dtype=float)
v = numpy.array([1,1,1,1,1,1,1],dtype=float)
lagra = numpy.array([1,2,0],dtype=float)
V = numpy.eye(7)

m = Model()

print('normal function evaluation')
m.eval_f(x)
m.eval_g(x)

print('Forward Mode')

grad_f_forward          = m.eval_grad_f_forward(x)
jac_g_forward           = m.eval_jac_g_forward(x)
jac_vec_g_forward       = m.eval_jac_vec_g_forward(x,[1,0,0,0,0,0,0])
grad_Lagrangian_forward = m.eval_grad_Lagrangian_forward(lam, x)
hess_Lagrangian_forward = m.eval_hess_Lagrangian_forward(lam, x)
vec_hess_g_forward      = m.eval_vec_hess_g_forward(lagra, x)

print(grad_f_forward)
print(jac_g_forward)
print(jac_vec_g_forward)
print(grad_Lagrangian_forward)
print(hess_Lagrangian_forward)
print(vec_hess_g_forward)

print('Reverse Mode')
m.trace_eval_f(x)
m.trace_eval_g(x)
grad_f_reverse           = m.eval_grad_f_reverse(x)
jac_g_reverse            = m.eval_jac_g_reverse(x)
hess_f_reverse           = m.eval_hess_f_reverse(x)
hess_vec_f_reverse       = m.eval_hess_vec_f_reverse(x,v)
vec_hess_g_reverse       = m.eval_vec_hess_g_reverse(lagra, x)
grad_Lagrangian_reverse  = m.eval_grad_Lagrangian_reverse(lam, x)
hess_Lagrangian_reverse  = m.eval_hess_Lagrangian_reverse(lam, x)

print(grad_f_reverse)
print(jac_g_reverse)
print(hess_f_reverse)
print(hess_vec_f_reverse)
print(vec_hess_g_reverse)
print(grad_Lagrangian_reverse)

from numpy.testing import assert_almost_equal
assert_almost_equal(grad_f_forward, grad_f_reverse)
assert_almost_equal(jac_g_forward, jac_g_reverse)
assert_almost_equal(grad_Lagrangian_forward, grad_Lagrangian_reverse)
assert_almost_equal(hess_Lagrangian_forward, hess_Lagrangian_reverse)
assert_almost_equal(vec_hess_g_forward, vec_hess_g_reverse)