import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import linalg
from scipy import optimize
import os
from scipy import interpolate
import scipy
###################################################################
def random_pattern(N,alpha,gaussian=True):
# Generating random matrix with mean-0, variance-1, size is N*alpha times N
    if gaussian:
        return np.random.randn(int(N*alpha),N)
    else:
        return np.sign(np.random.rand(int(N*alpha),N)-0.5)
###################################################################    
def X(N,alpha,c,d,gamma,tau=1,if_circulant=True,segment=1):
# Generating X matrix
# tau is the degree of asymmetry
# segement is the number of pattern sequences
    print("c={},d={},gamma={},tau={},segment={}".format(c,d,gamma,tau,segment))
    P = int(N*alpha)
    S = int(P/segment)
    for k in range(segment):
        if k==segment-1:
            dim=P-S*(segment-1)
        else:
            dim = S
        first_row = np.zeros(dim)
        first_col  =np.zeros(dim)
        first_row[0]=c
        first_col[0]=c
        for i in range(d):
            first_col[i+1] = (1-tau)*gamma
            first_row[i+1] =gamma
            if if_circulant:
                first_col[-(i+1)] = gamma
                first_row[-(i+1)] =(1-tau)*gamma
        block_matrix = linalg.toeplitz(first_row,first_col)
        if k==0:
            Xmatrix = block_matrix
        else:
            Xmatrix = scipy.linalg.block_diag(Xmatrix,block_matrix)        
    return Xmatrix
###################################################################    
def generate_weight(X,pattern,clean_diag = False,diluted=1):
# Generating connectivity
# clean_diag is to clear the diaganol elements
    N = np.shape(pattern)[1]
    J = 1.0/(N*diluted)*pattern.T@X@pattern*np.random.binomial(1,diluted,size=(N,N))
    if clean_diag:
        return (1-np.eye(N))*J
    else:
        return J
###################################################################    
def jacobi_weight(para,var=1.0,phip=lambda x: 1.0-np.tanh(x)**2,gaussian=False,if_circulant=True,if_ja=False,N=3000,segment=1,diluted=1):
    N=N
    alpha,c,d,gamma = para
    X_ = X(N,alpha,c,d,gamma,tau=1,if_circulant=if_circulant,segment=segment)
    pattern_ = random_pattern(N,alpha,gaussian=gaussian)
    J_ = generate_weight(X_,pattern_,diluted=diluted)
    input_current  =  np.sqrt(var)*np.random.randn(N)
    if if_ja:
        eigen_jacobi = np.linalg.eigvals(J_@np.diag(phip(input_current)))
    else:
        eigen_jacobi = np.linalg.eigvals(J_)
    return eigen_jacobi.real,eigen_jacobi.imag
####################################################################
def eigen(P,para):
# Generating eigenvalues of the X matrix
    alpha,c,d,gamma=para
    rn = np.linspace(0,1,num=P) 
    Lambda = (c + gamma*np.stack([np.exp(-2j*np.pi*rn*(i+1)) for i in range(d)],0).sum(0))
    return Lambda
#######################################################################
def curve_sorting(x,y,kind="linear",copy=True,interpolate=True,sorting=True):
    """sort the data in angle order"""
    # copy the another part of the spectrum
    if copy:
        x = np.append(x,x[::-1])
        y = np.append(y,-y[::-1])
    if not sorting:
        return x,y
    # clean the repeat elements
    zz = np.unique(np.array([x,y]),axis=1)
    x = zz[0]
    y=zz[1]
    # find the central of the data
    x0 = (np.max(x)+np.min(x))/2
    y0 = np.average(y)
    sx = x-x0
    sy = y-y0
    z = 1j*sy+sx
    angle = np.angle(z)
    # sort the data in angle order
    idx   = np.argsort(angle)
    x = np.array(x)[idx]
    y = np.array(y)[idx]
    angle = np.array(angle)[idx]
    if interpolate:
        fx = interpolate.interp1d(angle,x-x0,kind=kind,fill_value="extrapolate")
        fy = interpolate.interp1d(angle,y-y0,kind=kind,fill_value="extrapolate")
        new_angle = np.linspace(-np.pi,np.pi,num=1000,endpoint=True)
        new_x = fx(new_angle)+x0
        new_y = fy(new_angle)+y0    
        new_y[-1]=new_y[0]
        return new_x,new_y
    else:
        return x,y
