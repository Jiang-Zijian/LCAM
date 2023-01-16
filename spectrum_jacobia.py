import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import linalg
from scipy import optimize
import os
import tools

################################################################################################################################
"""
Marching method for the spectrum boundary
"""

def gradient_boundary(y,var,phip,lbd,alpha,real=False):
    """
    y is a 4-d real vector
    y[0]:Re(\hat{t}); y[1]:Im(\hat{t}); y[2]:Re(\hat{t}^\prime); y[3]: Im(\hat{t}^\prime)
    
    x is a 4-d complex vector
    x[0]:\hat{t}; x[1]:\hat{t}^\dagger; x[2]:(\hat{t}^\prime); x[3]: (\hat{t}^\prime)^\dagger
    
    calculate the gradient of three surfaces
    
    return value of the 3 equations [F1,F2,F3] and the 3x4 gradient matrix
        
    """
    def par_complex_to_real(x):
        return np.real(np.array([x[0]+x[1],1j*(x[0]-x[1]),x[2]+x[3],1j*(x[2]-x[3])]))
    if real==True:
        y = np.array([y[0],0,y[1],0])
    x = np.array([y[0]+1j*y[1],y[0]-1j*y[1],y[2]+1j*y[3],y[2]-1j*y[3]])
    def delta(a):
        if a==0:
            return np.array([1,0,0,0])
        if a==1:
            return np.array([0,1,0,0])
        if a==2:
            return np.array([0,0,1,0])
        if a==3:
            return np.array([0,0,0,1])
    def average1(z):
        t=x[0]
        tp=x[2]
        prob = 1.0/np.sqrt(2*np.pi*var)*np.exp(-z**2/(2*var))
        phipx = phip(z)
        return prob*1.0/(np.conjugate(phipx*tp+t)*(phipx*tp+t))

    def C():
        # C
        return integrate.quad(lambda z: average1(z)*(phip(z))**2,-1000,1000)[0]
    def C_():
        #dC/dx
        t=x[0]
        tp=x[2]
        def func1(z):return -1.0*(phip(z)**2)*average1(z)*(1.0/(phip(z)*tp+t))            
        def func2(z):return -1.0*(phip(z)**2)*average1(z)*(phip(z)/(phip(z)*tp+t))
        ga = integrate.quad(lambda z: np.real(func1(z)),-500,500)[0]
        gb = integrate.quad(lambda z: np.imag(func1(z)),-500,500)[0]
        gc = integrate.quad(lambda z: np.real(func2(z)),-500,500)[0]
        gd = integrate.quad(lambda z: np.imag(func2(z)),-500,500)[0]
        return np.array([ga+1j*gb,ga-1j*gb,gc+1j*gd,gc-1j*gd])
    def B():
        #B
        return integrate.quad(lambda z: average1(z)*(phip(z)),-1000,1000)[0]
    def B_():
        #dB/dx
        t=x[0]
        tp=x[2]
        def func1(z):return -1.0*(phip(z))*average1(z)*(1.0/(phip(z)*tp+t))            
        def func2(z):return -1.0*(phip(z))*average1(z)*(phip(z)/(phip(z)*tp+t))
        ga = integrate.quad(lambda z: np.real(func1(z)),-500,500)[0]
        gb = integrate.quad(lambda z: np.imag(func1(z)),-500,500)[0]
        gc = integrate.quad(lambda z: np.real(func2(z)),-500,500)[0]
        gd = integrate.quad(lambda z: np.imag(func2(z)),-500,500)[0]
        return np.array([ga+1j*gb,ga-1j*gb,gc+1j*gd,gc-1j*gd])

    
    CC = C()
    BB = B()
    GG = BB*x[0]+CC*x[2]
    CC_ = C_()
    BB_ = B_()
    
    def G1_():
        #dG^\prime/dx
        return BB_*x[0]+delta(0)*BB+CC_*x[2]+delta(2)*CC
    def G2_():
        #dG^\prime^\dagger /dx
        return BB_*x[1]+delta(1)*BB+CC_*x[3]+delta(3)*CC
         
    def kI():        
        return np.average((np.conjugate(lbd)*lbd)/(np.conjugate(lbd*GG-1)*(lbd*GG-1)))
    def kI_part():
        return (np.conjugate(lbd)*lbd)/(np.conjugate(lbd*GG-1)*(lbd*GG-1))
    def kIp(): 
        return np.average((lbd)/(np.conjugate(lbd*GG-1)*(lbd*GG-1)))
    def kIp_part():
        return (lbd)/(np.conjugate(lbd*GG-1)*(lbd*GG-1))
    def kI1_():    
        """
        d kI/d G^\prime
        """
        return np.average(-1.0*kI_part()*lbd/(lbd*GG-1))
    def kI2_():    
        """
        d kI/d (G^\prime)^\dagger
        """
        return np.conjugate(kI1_())
    def kI3_():    
        """
        d kI^\prime/d G^\prime
        """
        return np.average(-1.0*kIp_part()*lbd/(lbd*GG-1))
    def kI4_():    
        """
        d kI^\prime /d (G^\prime)^\dagger
        """
        return np.average(-1.0*kIp_part()*np.conjugate(lbd/(lbd*GG-1)))
    
    def kI5_():    
        """
        d (kI^\prime)^\dagger /d G^\prime
        """
        return np.conjugate(kI4_())
    def kI6_():    
        """
        d (kI^\prime)^\dagger /d (G^\prime)^\dagger
        """
        return np.conjugate(kI3_())

    def GG1_():
        """
        dF1/dx
        """
        g = alpha*(CC_*kI()+CC*kI1_()*G1_()+CC*kI2_()*G2_())
        return g
    def GG2_():
               
        """
        dF2/dx
        """
        g = alpha*(CC_*kIp()+CC*kI3_()*G1_()+CC*kI4_()*G2_())-BB_*x[1]-BB*delta(1)
        return g
    def GG3_():
        """
        dF3/dx
        """
        g = alpha*(CC_*np.conjugate(kIp())+CC*kI5_()*G1_()+CC*kI6_()*G2_())-BB_*x[0]-BB*delta(0)
        return g
    g1 = GG1_()
    g2a = GG2_()
    if real==False:
        g2b = GG3_()
        g2 = 0.5*(g2a+g2b)
        g3 = 1.0/2j*(g2a-g2b)
    f1 = alpha*CC*kI()-1
    f2a = alpha*CC*kIp()-BB*x[1]
    if real==False:
        f2b = alpha*CC*np.conjugate(kIp())-BB*x[0]
        f2 = 0.5*(f2a+f2b)
        f3 = 1.0/(2j)*(f2a-f2b)
    if real==True:
        ff = np.real(np.array([f1,f2a]))
        g11 = [par_complex_to_real(g1)[0],par_complex_to_real(g1)[2]]
        g22 = [par_complex_to_real(g2a)[0],par_complex_to_real(g2a)[2]]
        return ff, np.array([g11,g22])
    return np.real(np.array([f1,f2,f3])),np.array([par_complex_to_real(g1),par_complex_to_real(g2),par_complex_to_real(g3)])
def newton_step(f,g):
    A = g@g.T
    if np.abs(np.linalg.det(A))>1e-10:
        Delta = -np.linalg.inv(A)@f
    else:
        return np.nan
    eq_num = g.shape[0]
    var_num = g.shape[1]
    Delta = np.reshape(Delta,(eq_num,1))
    Delta = np.tile(Delta,var_num)
    
    return np.sum(Delta*g,axis=0)
def march_boundary(y0,var,phip,lbd,alpha,lr=0.1,if_print=False,jump=True):
    f,g=gradient_boundary(y0,var,phip,lbd,alpha)
#     print(f)
    y= np.array(y0)
    for i in range(200):
#         print(np.sum(np.abs(f)))
        if np.sum(np.abs(f))>0.3*1e-2:
            if if_print:
                print("Times: {} |F:{} y:{}".format(i,f,y))
            dy = newton_step(f,g)
            if np.sum(np.abs(f))<1e-1:
                if jump:
                    lr = 1.0
            y = y+lr*dy
            f,g=gradient_boundary(y,var,phip,lbd,alpha)
        else:
            return y
    print("not converge in 200 interations")
    return np.nan
def tangent(g,option=1):
    """
    option = 1: scale the tangent step by the first two variables
    option = 2: scale the tangent step by the last two variables
    option = 3: scale the tangent step by all four variables
    """
    e1 = np.array([1,0,0,0])
    e2 = np.array([0,1,0,0])
    e3 = np.array([0,0,1,0])
    e4 = np.array([0,0,0,1])
    a1 = (g.T)[0]
    a2 = (g.T)[1]
    a3 = (g.T)[2]
    a4 = (g.T)[3]
    A1 = np.linalg.det((np.array([a2,a3,a4])).T)
    A2 = np.linalg.det((np.array([a1,a3,a4])).T)
    A3 = np.linalg.det((np.array([a1,a2,a4])).T)
    A4 = np.linalg.det((np.array([a1,a2,a3])).T)
    T = A1*e1-A2*e2+A3*e3-A4*e4

    length1 = np.sqrt(T[0]**2+T[1]**2)
    length2 = np.sqrt(T[2]**2+T[3]**2)
    length3 = np.sqrt(np.sum(T**2))
    length = [length1,length2,length3]
    return T/length[option-1]
def march_method(y0,var,phip,lbd,alpha,lr=0.15,step_len = 0.05,if_print=False,option=1,num=100,auto=False):
    all_y=[]
    step_len_old = step_len
    y_init = y0
    if auto:
        num = 300
        flag1 = 0
        flag2 = 0
    angle_record=[] 
    y_record=[]
    all_y.append([y0[0],0,y0[2],0])
    y_record.append(0)
    for i in range(1,num):
        
        y = march_boundary(y0,var,phip,lbd,alpha,lr=lr,if_print=if_print)
        all_y.append(y)
        y_record.append(np.abs(y[1]))
        if i>=2:
            lr =0.4
        if i>=5:          
            if np.abs(y[1])<1.0/4*max(y_record):
                step_len = 1.0/5*step_len_old
            else:
                step_len=step_len_old
        if auto:
            if i>=10:
                if (y[1]-y_init[1])*(all_y[i-1][1]-y_init[1])<0:
                        return np.array(all_y)
                if np.abs(y[1]-y_init[1])<step_len_old:
                        return np.array(all_y)
            
            
        
        if i%20==0:
            print("Point {}: {}".format(i,y))
        
        f,g = gradient_boundary(y,var,phip,lbd,alpha)
        step = tangent(g,option=option)
        y0 = y+step_len*step
        
            
    return np.array(all_y)
def single_start_point(y0,var,phip,lbd,alpha,lr=0.1,ifprint=True):
    y = y0
    for i in range(40):
        f,g = gradient_boundary(y0,var,phip,lbd,alpha,real=True)
        y0 = y0+lr*newton_step(f,g)
        if np.sum(np.abs(f))<8e-1:
            lr = 1.0
        if np.isnan(f[0]):
            raise RuntimeError("Error!!!")
        if np.sum(np.abs(f))>1e2:
            raise RuntimeError("Diverse!!!")
        if ifprint:
            print(f,y0)
        if np.sum(np.abs(f))<1e-4:
            print("converge in {}".format(i+1))
            return y0
    print("failure")
    raise RuntimeError("failure in 200 iterlations")
def sort_different_root(record):
    dif_roots  = []
    flag = 0
    for i in range(len(record)):
        if flag==0 and (not np.isnan(record[i][0])):
            dif_roots.append(record[i])
            flag = 1
    for i in range(len(record)):
        if not np.isnan(record[i][0]):
            same = 0
            for root in dif_roots:
                if np.sum(abs(record[i]-root))<1e-2:
                    same = 1
            if same ==0:
                dif_roots.append(record[i])
    dif_roots = np.array(dif_roots)  
    order = np.argsort(dif_roots,axis=0)
    return dif_roots[order[:,0]]
def draw_all(var,phip,lbd,alpha,lr=0.1,num=50,auto=False,step_len = 0.15,onlyinitpoint=False,scannum=30):
    print("Scanning for the initial points")
    xxx = np.linspace(-1.5,1.5,num=scannum)
    yyy = 1*xxx
    xxx = xxx.reshape(-1)
    yyy= yyy.reshape(-1)
    N = len(xxx)
    record = 0*np.ones_like(xxx)
    roots = []
    for i in range(N):
        y0 = [xxx[i],yyy[i]]
        try:
            result = single_start_point(y0,var,phip,lbd,alpha,ifprint=False)
            
        except:
            result = [np.nan,np.nan]
        roots.append(result)
        print("total: {}/{} result: {}".format(i+1,N,result))
    dif_roots = sort_different_root(roots)
    for i in range(N):
        for k in range(len(dif_roots)):
            if np.sum(np.abs(roots[i]-dif_roots[k]))<1e-2:
                record[i] = k+1
    print("the initial points are "+str(dif_roots))
    if onlyinitpoint:
        return np.array(dif_roots)
    all_trajectories = []
    for root in dif_roots:
        y0 = np.array([root[0],0.01,root[1],0.01])
        all_trajectories.append(march_method(y0,var,phip,lbd,alpha,lr=lr,step_len = step_len,if_print=False,option=1,num=num,auto=auto))
    return all_trajectories


####################################################################################################################################
"""
MARCHING METHOD for density
"""

def gradient_density(y,where,var,phip,lbd,alpha,real=False):
    """
    y is a 3-d real vector
    y[0]:Re(\hat{t}^\prime); y[1]:Im(\hat{t}^\prime); y[2]:\hat{u}^\prime \hat{v}; 
    x is a 3-d complex vector
    x[0]:\hat{t}^\prime; x[1]:(\hat{t}^\prime)^\dagger; x[2]:\hat{u}^\prime \hat{v}
    "where" is the location on complex plane
    calculate the gradient of three surfaces
    
    return value of the 3 equations [F1,F2,F3] and the 3x3 gradient matrix
        
    """
    def par_complex_to_real(x):
        return np.real(np.array([x[0]+x[1],1j*(x[0]-x[1]),x[2]]))
    
    x = np.array([y[0]+1j*y[1],y[0]-1j*y[1],y[2]])
    t = np.conjugate(where)
    def delta(a):
        # modified
        if a==0:
            return np.array([1,0,0])
        if a==1:
            return np.array([0,1,0])
        if a==2:
            return np.array([0,0,1])
        if a==3:
            return np.array([0,0,0])
    def average1(z):
        tp=x[0]
        uvh = x[2]
        prob = 1.0/np.sqrt(2*np.pi*var)*np.exp(-z**2/(2*var))
        phipx = phip(z)
        return prob*1.0/(np.conjugate(phipx*tp+t)*(phipx*tp+t)-phipx**2*uvh)
    def q(z):
        tp=x[0]
        uvh = x[2]
        phipx = phip(z)
        return 1.0/(np.conjugate(phipx*tp+t)*(phipx*tp+t)-phipx**2*uvh)
        
    def C():
        # C
        return integrate.quad(lambda z: average1(z)*(phip(z))**2,-1000,1000)[0]
    def C_():
        #dC/dx
        tp=x[0]
        uvh = x[2]
        def func1(z):return -1.0*(phip(z)**2)*average1(z)*q(z)*phip(z)*np.conjugate(phip(z)*tp+t)            
        def func2(z):return -1.0*(phip(z)**2)*average1(z)*q(z)*(-1.0*phip(z)**2)
        ga = integrate.quad(lambda z: np.real(func1(z)),-500,500)[0]
        gb = integrate.quad(lambda z: np.imag(func1(z)),-500,500)[0]
        gc = integrate.quad(lambda z: np.real(func2(z)),-500,500)[0]
        return np.array([ga+1j*gb,ga-1j*gb,gc])
    def B():
        #B
        return integrate.quad(lambda z: average1(z)*(phip(z)),-1000,1000)[0]
    def A():
        #A
        return integrate.quad(lambda z: average1(z),-1000,1000)[0]
    def B_():
        #dB/dx
        tp=x[0]
        uvh = x[2]
        def func1(z):return -1.0*(phip(z))*average1(z)*q(z)*phip(z)*np.conjugate(phip(z)*tp+t)            
        def func2(z):return -1.0*(phip(z))*average1(z)*q(z)*(-1.0*phip(z)**2)
        ga = integrate.quad(lambda z: np.real(func1(z)),-500,500)[0]
        gb = integrate.quad(lambda z: np.imag(func1(z)),-500,500)[0]
        gc = integrate.quad(lambda z: np.real(func2(z)),-500,500)[0]
        return np.array([ga+1j*gb,ga-1j*gb,gc])

    AA = A()
    CC = C()
    BB = B()
    GG = BB*t+CC*x[0]
    UV = -CC**2*x[2]
    CC_ = C_()
    BB_ = B_()
    
    def G1_():
        #dG^\prime/dx
        return BB_*t+delta(3)*BB+CC_*x[0]+delta(0)*CC
    def G2_():
        #dG^\prime^\dagger /dx
        return BB_*np.conjugate(t)+delta(3)*BB+CC_*x[1]+delta(1)*CC
    def uv_():
        # d u^\prime v / dx
        return -2*CC*CC_*UV-CC**2*delta(2)
    
    def kI():        
        return np.average((np.conjugate(lbd)*lbd)/(np.conjugate(lbd*GG-1)*(lbd*GG-1)+UV*np.conjugate(GG)*GG))
    def kI_part():
        return (np.conjugate(lbd)*lbd)/(np.conjugate(lbd*GG-1)*(lbd*GG-1)+UV*np.conjugate(GG)*GG)**2
    def kIp(): 
        return np.average((lbd)/(np.conjugate(lbd*GG-1)*(lbd*GG-1)+UV*np.conjugate(GG)*GG))
    def kIp_part():
        return (lbd)/(np.conjugate(lbd*GG-1)*(lbd*GG-1)+UV*np.conjugate(GG)*GG)**2
    def kI1_():    
        """
        d kI/d G^\prime
        """
        return np.average(-1.0*kI_part()*(lbd*np.conjugate(lbd*GG-1)+UV*np.conjugate(GG)))
    def kI2_():    
        """
        d kI/d (G^\prime)^\dagger
        """
        return np.conjugate(kI1_())
    def kI3_():    
        """
        d kI^\prime/d G^\prime
        """
        return np.average(-1.0*kIp_part()*(lbd*np.conjugate(lbd*GG-1)+UV*np.conjugate(GG)))
    def kI4_():    
        """
        d kI^\prime /d (G^\prime)^\dagger
        """
        return np.average(-1.0*kIp_part()*(np.conjugate(lbd)*(lbd*GG-1)+UV*(GG)))
    
    def kI5_():    
        """
        d (kI^\prime)^\dagger /d G^\prime
        """
        return np.conjugate(kI4_())
    def kI6_():    
        """
        d (kI^\prime)^\dagger /d (G^\prime)^\dagger
        """
        return np.conjugate(kI3_())
    def kI11_():
        """
        d (kI) /d u^\prime v
        """
        return np.average(-1.0*kI_part()*(1.0*np.conjugate(GG)*GG))
    def kI22_():
        """
        d (kI^\prime) /d u^\prime v
        """
        return np.average(-1.0*kIp_part()*(1.0*np.conjugate(GG)*GG))
    def kI33_():
        return np.conjugate(kI22_())
                          
    def GG1_():
        """
        dF1/dx
        """
        g = alpha*(CC_*kI()+CC*kI1_()*G1_()+CC*kI2_()*G2_()+CC*kI11_()*uv_())
        return g
    def GG2_():
               
        """
        dF2/dx
        """
        g = alpha*(CC_*kIp()+CC*kI3_()*G1_()+CC*kI4_()*G2_()+CC*kI22_()*uv_())-BB_*np.conjugate(t)-BB*delta(3)
        return g
    def GG3_():
        """
        dF3/dx
        """
        g = alpha*(CC_*np.conjugate(kIp())+CC*kI5_()*G1_()+CC*kI6_()*G2_()+CC*kI33_()*uv_())-BB_*t-BB*delta(3)
        return g
    g1 = GG1_()
    g2a = GG2_()
    if real==False:
        g2b = GG3_()
        g2 = 0.5*(g2a+g2b)
        g3 = 1.0/2j*(g2a-g2b)
    f1 = alpha*CC*kI()-1
    f2a = alpha*CC*kIp()-BB*np.conjugate(t)
    if real==False:
        f2b = alpha*CC*np.conjugate(kIp())-BB*t
        f2 = 0.5*(f2a+f2b)
        f3 = 1.0/(2j)*(f2a-f2b)
    GG_need = AA*t+BB*x[0]
    return np.real(np.array([f1,f2,f3])),np.array([par_complex_to_real(g1),par_complex_to_real(g2),par_complex_to_real(g3)]),GG_need

    
# def newton_step(f,g):
#     A = g@g.T
#     if np.abs(np.linalg.det(A))>1e-10:
#         Delta = -np.linalg.inv(A)@f
#     else:
#         return np.nan
#     eq_num = g.shape[0]
#     var_num = g.shape[1]
#     Delta = np.reshape(Delta,(eq_num,1))
#     Delta = np.tile(Delta,var_num)
    
#     return np.sum(Delta*g,axis=0)
def march_density(y0,where,var,phip,lbd,alpha,lr=0.1,if_print=False,jump=True):
    f,g,G=gradient_density(y0,where,var,phip,lbd,alpha)
#     print(f)
    y= np.array(y0)
    for i in range(200):
#         print(np.sum(np.abs(f)))
        if np.sum(np.abs(f))>0.3*1e-5:
            if if_print:
                print("Times: {} |F:{} y:{} G:{}".format(i,f,y,G))
            dy = newton_step(f,g)
            if np.sum(np.abs(f))<1e-1:
                if jump:
                    lr = 1.0
            y = y+lr*dy
            f,g,G=gradient_density(y,where,var,phip,lbd,alpha)
        else:
            return y, G
    print("not converge in 200 interations")
    return np.nan
#######################
def density(y0,where,var,phip,lbd,alpha,if_print=True,lr=0.3):
    dw = 0.001
    where1 = where+dw
    where2 = where-dw
    where3 = where+dw*1j
    where4 = where-dw*1j
    wheres = [where1,where2,where3,where4]
    Gs = []
    for i,w in enumerate(wheres):
        if i==0:
            y,G = march_density(y0,w,var,phip,lbd,alpha,if_print=if_print,lr=lr)
        else: 
            y,G = march_density(y,w,var,phip,lbd,alpha,if_print=if_print,lr=lr)
        print(G)
        Gs.append(G)
    dx = np.real(Gs[0]-Gs[1])/(2*dw)
    dy = np.imag(Gs[2]-Gs[3])/(2*dw)
    return 0.5/np.pi*(dx-dy),y
def numerical_density(var,phip,para,num_mat=100,dim_mat = 2000):
    all_x = []
    all_y = []
    for i in range(num_mat):
        if i%10==0:
            print(str(i)+" has finished")
        x,y = weight(var,phip,para=para,if_ja=True,num=dim_mat)
        all_x.append(x)
        all_y.append(y)
        
    all_x = np.reshape(np.array(all_x),-1)
    all_y = np.reshape(np.array(all_y),-1)
    a = plt.hist2d(all_x,all_y,bins=50,normed=True)
    Z = a[0]
    x = a[1]
    y =a[2]
    x_middle = []
    y_middle = []
    for i in range(len(x)-1):
        x_middle.append((x[i]+x[i+1])/2)
        y_middle.append((y[i]+y[i+1])/2)
        
    
    X,Y = np.meshgrid(y_middle,x_middle)
    return np.array(x_middle),np.array(y_middle),Z
        

    