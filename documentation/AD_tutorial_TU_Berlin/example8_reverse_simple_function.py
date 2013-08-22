import numpy; from numpy import sin,cos; from taylorpoly import UTPS
x1 = UTPS([3,1,0],P=2); x2 = UTPS([7,0,1],P=2)
# forward mode 
vm1 = x1
v0  = x2
v1 = cos(v0)
v2 = v1 * vm1
v3 = vm1 + v2
v4 = sin(v3)
y  = v4
# reverse mode
v4bar = UTPS([0,0,0],P=2);  v3bar = UTPS([0,0,0],P=2)
v2bar = UTPS([0,0,0],P=2);  v1bar = UTPS([0,0,0],P=2)
v0bar = UTPS([0,0,0],P=2); vm1bar = UTPS([0,0,0],P=2)
v4bar.data[0] = 1.
v3bar += v4bar*cos(v3) 
vm1bar += v3bar; v2bar += v3bar
v1bar += v2bar * vm1;  vm1bar += v2bar * v1
v0bar -= v1bar * sin(v0) 
g1 = y.data[1:]; g2 = numpy.array([vm1bar.data[0], v0bar.data[0]])
print('UTPS gradient g(x_0)=', g1)
print('reverse gradient g(x_0)=', g2)
print('Hessian H(x_0)=\n',numpy.vstack([vm1bar.data[1:], v0bar.data[1:]]))
