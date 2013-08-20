import numpy

def mul(D,x_data, y_data, z_data):
    for d in range(D):
        for k in range(d+1):
            z_data[d] += x_data[k] * y_data[d-k]
            
    return z_data

def pbr_mul(D, x_data, y_data, z_data, zbar_data, xbar_data, ybar_data):
    for d in range(D):
        for k in range(D-d):
            xbar_data[d] += zbar_data[d+k] * y_data[k]
            ybar_data[d] += zbar_data[d+k] * x_data[k]
            
    return (xbar_data, ybar_data)
            
            
if __name__ == "__main__":
    """ compute z = x*x*x
    """
    
    x = numpy.array([5.,1.])
    
    # forward sweep
    y = mul(2, x, x, numpy.zeros(2))
    z = mul(2, y, x, numpy.zeros(2))
    
    # reverse sweep
    zbar = numpy.array([0.,1.])
    ybar = numpy.zeros(2)
    xbar = numpy.zeros(2)
    
    pbr_mul(2, y, x, z, zbar, ybar, xbar)
    pbr_mul(2, x, x, y, ybar, xbar, xbar)
    
    print(x)
    print(xbar)
    
    
    

