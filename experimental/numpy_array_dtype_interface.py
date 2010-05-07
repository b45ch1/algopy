import numpy

class oofun:
    is_dtype = True

    def __init__(self, data):
        self.data = numpy.asarray(data,dtype=float)

    def __mul__(self, other):
        return oofun(oofun.multiply(self.data, other.data))
        
    
    @classmethod
    def spsd_multiply(cls, x_data, y_data, z_data = None):
        """
        Single Program Single Data implementation can only do one data type multiplication at a time
        
        raw algorithm that operates on possibly nested containers (array,list,tuple,dict) of elementary Python data types: float, int, ...
        
        as an example, we use here numy.array of shape = (3,) with dtype float for x_data, y_data, z_data

        if wanted, memory allocation of z_data can be done before calling this function
        (would not be of use here, since there is no convolve function where the output array can be preallocated)

        """

        z_data = x_data * y_data
        return z_data

    @classmethod
    def spmd_multiply(cls, array_of_x_data, array_of_y_data, array_of_z_data = None):
        """
        Single Program Multiple Data implementation is responsible for efficient vectorized multiplication

        array_of_x_data is an array of shape = (N1,N2,...) + (3,)

        """

        array_of_z_data = array_of_x_data * array_of_y_data
        return array_of_z_data
    

    def __str__(self):
        return str(self.data) + 'a'


class array:
    def __init__(self, input_list, dtype=float):
        if dtype != float:
            
            


x = oofun([1,2,3])
y = oofun([4,5,6])

z = x * y

print x
print y
print z

x_array = array([oofun([1,2,3]),oofun([4,5,6]])],dtype=oofun)
y_array = array([oofun([7,8,9]),oofun([3,5,1]])],dtype=oofun)

