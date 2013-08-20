import sympy
x,y,z = sympy.symbols('xyz')

def f(x,y,z):
    return x*y * z*x - y + x*(z-x*y)
    
u = f(x,y,z)
print('f(x,y,z) = ',u)
g = [u.diff(x), u.diff(y), u.diff(z)]
print('grad f(x,y,z) =', g)
