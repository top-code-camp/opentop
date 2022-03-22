# collecting useful functions in topology optimization

# simp
def simp(x,p,xmin,xmax):
    return xmin + x**p * (xmax-xmin)    

# derivatives of simp
def dsimp(x,p,xmin,xmax):
    return (p*(xmax-xmin)) * x**(p-1)
       
def projection():
    print('projection is not defined')

def dprojection():
    print('dprojection is not defined')
    
def filter():
    print('filter is not defined')

def dfilter():
    print('dfilter is not defined')
    
def oc():
    print('oc is not defined')

# example of using function handles
def mat_interpolation(model=None,xmin=1e-9,xmax=1):
    if model is None:
        model = 'simp'
    if model == 'simp':
        E = lambda x, p : simp(x,p,xmin,xmax)
        dE = lambda x, p : dsimp(x,p,xmin,xmax)
        return E,dE