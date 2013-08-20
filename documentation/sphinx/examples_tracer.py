
from algopy import CGraph, Function
cg = CGraph()
cg.trace_on()
x = Function(1)
y = Function(3)
z = x * y + x
cg.trace_off()
cg.independentFunctionList = [x,y]
cg.dependentFunctionList = [z]
print(cg)
cg.plot('example_tracer_cgraph.png')