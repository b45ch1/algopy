import numpy, algopy
from algopy import UTPM, exp

def eval_f(x):
    """ some function """
    return x[0]*x[1]*x[2] + exp(x[0])*x[1]

# forward mode without building the computational graph
# -----------------------------------------------------
x = UTPM.init_jacobian([3,5,7])
y = eval_f(x)
algopy_jacobian = UTPM.extract_jacobian(y)
print('jacobian = ',algopy_jacobian)

# reverse mode using a computational graph
# ----------------------------------------

# STEP 1: trace the function evaluation
cg = algopy.CGraph()
x = algopy.Function([1,2,3])
y = eval_f(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]

# STEP 2: use the computational graph to evaluate derivatives
print('gradient =', cg.gradient([3.,5,7]))
print('Jacobian =', cg.jacobian([3.,5,7]))
print('Hessian =', cg.hessian([3.,5.,7.]))
print('Hessian vector product =', cg.hess_vec([3.,5.,7.],[4,5,6]))



