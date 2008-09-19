#!/usr/bin/env python


fileHandle = open ( 'test.txt', 'w' ) 
varcount = 0



class adouble:
	def __init__(self,x,dx = 0):
		global varcount
		fileHandle.write('double x%d = %f;\ndouble dx%d = %f;\n'%(varcount,x, varcount, dx))
		self.locid = varcount
		varcount +=1
		self.x = x
		self.dx = dx
	def __mul__(self,rhs):
		retval = adouble(self.x * rhs.x, self.x * rhs.dx + self.dx * rhs.x)
		fileHandle.writelines('x%d = x%d * x%d;\n'%(retval.locid,self.locid,rhs.locid))
		fileHandle.writelines('dx%d = dx%d * x%d + x%d * dx%d;\n'%(retval.locid,self.locid, rhs.locid,self.locid, rhs.locid))
		return retval
	def __str__(self):
		return '[%f,%f]'%(self.x,self.dx)


if __name__ == "__main__":
	x = adouble(2,1)
	y = adouble(3)
	print x*y
	
fileHandle.close()