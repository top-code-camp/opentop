import numpy as np

# Unit: m,Pa,s
def shape_function(s, t):
    N1 = (1 - s) * (1 - t) / 4
    N2 = (1 + s) * (1 - t) / 4
    N3 = (1 + s) * (1 + t) / 4
    N4 = (1 - s) * (1 + t) / 4
    return np.array([[N1, 0, N2, 0, N3, 0, N4, 0], [0, N1, 0, N2, 0, N3, 0, N4]])
    
    
def constitutive_matrix(E,v,problem = None):

    if problem is None:
        problem = 'plane_strain'
        
    if problem == 'plane_strain':
        # plane strain problem
        return np.array([[1-v, v, 0],[v,1-v,0],[0,0,(1-v)/2]])*(E/(1-v**2))
        
    if problem == 'plane_stress':
        # add plane stress problem below
        return np.array([[1-v, v, 0],[v,1-v,0],[0,0,0.5-v]])*(E/(1+v)/(1-2*v))

if __name__ == "__main__":
    # test all functions in this module
    N = shape_function(1,1)
    print('N = \n',N)
    D = constitutive_matrix(1,0.3,'plane_stress')
    print('D = \n',D)
