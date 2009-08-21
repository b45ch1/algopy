from numpy import *
N = 10
M = 3
x = zeros(N)

ops = {
	'imul': 1,
}


class BB_tape:
	def __init__(self, bb_id):
		self.id = bb_id
		self.op_tape   = []
		self.loc_tape  = []
	def __str__(self):
		return str(self.op_tape) + str(self.loc_tape)
	
	def __repr__(self):
		return self.__str__()

i = 0
curr_bb = 0
bb_tapes = [BB_tape(i) for i in range(10)]


#class BB:
	#def __init__(self, id):
		#global curr_bb
		#self.id = id
		#curr_bb = id
	#def __str__(self):
		#return 'b'+str(self.id)
	#def __repr__(self):
		#return self.__str__()

class adouble:
	def __init__(self,x):
		global i
		self.x = x
		self.id = i
		i += 1
	def __str__(self):
		return str(self.id)
		
	def __imul__(self,rhs):
		bb_tapes[curr_bb].op_tape.append(ops['imul'])
		bb_tapes[curr_bb].loc_tape.append((self.id, rhs.id))
		self.x *= rhs.x
		return self

ax = adouble(2.)
ay = adouble(2.)


for n in range(N):
	curr_bb = 0
	ax *= ay
	
print bb_tapes
