import numpy; from numpy import sin,cos; import adolc
def f(x):
    return sin(x[0] + cos(x[1])*x[0])

adolc.trace_on(1)
x = adolc.adouble([3,7]);  adolc.independent(x)
y = f(x)
adolc.dependent(y); adolc.trace_off()
adolc.tape_to_latex(1,[3,7],[0.])

