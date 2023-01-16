import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import linalg
from scipy import optimize
import os
from scipy import interpolate
import tools
def F(x,lbd,para):
    alpha,c,d,gamma=para
    G = x[0]+1j*x[1]
    Lambda = lbd
    LG = Lambda*G
    LG_dagger = np.conjugate(LG) 
    return np.real(alpha*np.average(LG_dagger*LG/((LG_dagger-1)*(LG-1)))-1)
def F1(x,lbd,para):
    alpha,c,d,gamma=para
    G = x[0]+1j*x[1]
    Lambda = lbd
    LG = Lambda*G
    LG_dagger = np.conjugate(LG)
    part1 = np.conjugate(Lambda)*Lambda/((LG_dagger-1)*(LG-1))
    part2 = np.conjugate(G)/(LG-1)+G/(LG_dagger-1)
    return np.real(-alpha*np.average(part1*part2))
def F2(x,lbd,para):
    alpha,c,d,gamma=para
    G = x[0]+1j*x[1]
    Lambda = lbd
    LG = Lambda*G
    LG_dagger = np.conjugate(LG)
    part1 = np.conjugate(Lambda)*Lambda/((LG_dagger-1)*(LG-1))
    part2 = 1j*np.conjugate(G)/(LG-1)-1j*G/(LG_dagger-1)
    return np.real(-alpha*np.average(part1*part2))
def FF(x,lbd,para):
    """the second equation"""
    alpha,c,d,gamma=para
    G = x[0]+1j*x[1]
    lbd_dagger = np.conjugate(lbd)
    lbd_G = lbd*G
    lbd_G_dagger = np.conjugate(lbd_G)
    w = alpha*np.average(lbd/((lbd_G_dagger-1)*(lbd_G-1)))
    return np.array([w.real,w.imag])
def gradient(x,lbd,para):
    F_ = F(x,lbd,para)
    F1_ = F1(x,lbd,para)
    F2_ = F2(x,lbd,para)
    return np.reshape(F_,[1,1]),np.reshape(np.array([F1_,F2_]),[1,2])
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
def march_boundary(x0,lbd,para,lr=0.1,if_print=False,jump=True):
    f,g=gradient(x0,lbd,para)
    x= np.array(x0)
    for i in range(200):
        if np.sum(np.abs(f))>0.3*1e-4:
            if if_print:
                print("Times: {} |F:{} x:{}".format(i,f,x))
            dx = newton_step(f,g)
            if np.sum(np.abs(f))<5e-2:
                if jump:
                    lr = 1.0
            x = x+lr*dx
            f,g=gradient(x,lbd,para)
        else:
            return x
    print("not converge in 200 interations")
    return np.nan
def tangent(g):    
    """
    scale the tangent step by the first two variables
    """
    e1 = np.array([1,0])
    e2 = np.array([0,1])
    a1 = (g.T)[0]
    a2 = (g.T)[1]
    A1 = np.linalg.det((np.array([a2])).T)
    A2 = np.linalg.det((np.array([a1])).T)
    T = A1*e1-A2*e2
    length1 = np.sqrt(T[0]**2+T[1]**2)
    return T/length1
def sort_different_root(record):
    dif_roots  = []
    flag = 0
    for i in range(len(record)):
        if flag==0 and (not np.isnan(record[i])):
            dif_roots.append(record[i])
            flag = 1
    for i in range(len(record)):
        if not np.isnan(record[i]):
            same = 0
            for root in dif_roots:
                if np.sum(abs(record[i]-root))<1e-2:
                    same = 1
            if same ==0:
                dif_roots.append(record[i])
    dif_roots = np.array(dif_roots)  
    order = np.argsort(dif_roots,axis=0)
    return dif_roots[order]
def initial_point_finder(xmin,xmax,para,step=0.05):
    lbd = tools.eigen(para=para,P=100000)
    def func(x):
        return F([x,0],lbd,para)
    xs = np.linspace(xmin,xmax,num=int((xmax-xmin)/step),endpoint=True)
    all_y = []
    all_solution = []
    for x in xs:
        all_y.append(func(x))
    for i in range(len(xs)-1):
        if all_y[i]*all_y[i+1]<0:
            root=optimize.bisect(func,xs[i],xs[i+1])
            all_solution.append(root)
    return sort_different_root(np.array(all_solution))

def single_draw(x0,para,lr=0.1,step=0.05,num=200):
    step0=step
    lbd = tools.eigen(para=para,P=100000)
    all_x =[]
    all_w = []
    w = FF(x0,lbd,para)
    all_x.append(x0)
    all_w.append(w)
    x  = x0+np.array([0,0.01])
    for i in range(num):
        try:
            x = march_boundary(x,lbd,para,lr = lr)
            all_x.append(x)
            w = FF(x,lbd,para)
            all_w.append(w)
            print("Point:{}|\t x:{}|\t w:{}".format(i,x,w))
            if np.sqrt(np.sum(w-all_w[i])**2)>step0/2:
                step=step0/2
            else:
                step=step0
            print(np.sqrt(np.sum(w-all_w[i])**2),step)
            f,g = gradient(x,lbd,para)
            dx = tangent(g)
            if i<30:
                step=0.02
            x = x+step*dx
        except:
            pass
    return np.array(all_x),np.array(all_w)
def draw_all(para,lr=0.1,step=0.05,num=200):
    startpoints = initial_point_finder(-3,3,para,step=step)
    print(startpoints)
    all_xx = []
    all_ww = []
    for i in range(len(startpoints)):
        x0 = np.array([startpoints[i],0])
        all_x,all_w = single_draw(x0,para,lr=0.1,step=0.05,num=num)
        all_xx.append(all_x)
        all_ww.append(all_w)
    return all_xx,all_ww
# def spectrum_right_boundary_finder(x0=-3,x1=3,step=0.05,para=[1.5,1,1,1]):
#     Gs = initial_point_finder(xmin=x0,xmax=x1,step=step,para=para)
#     all_boundary=[]
#     lbd = tools.eigen(P=100000,para=para)
#     for G in Gs:
#         all_boundary.append(FF([G.real,G.imag],para=para,lbd=lbd)[0])
#     index = np.argmax(np.array(all_boundary))
#     print("boundary:{},G:{}".format(all_boundary[index],Gs[index]))
#     return all_boundary[index],Gs[index]

def spectrum_right_boundary_finder(x0,x1,step,para=[1.5,1,1,1]):
    Gs = initial_point_finder(xmin=x0,xmax=x1,step=step,para=para)
    all_boundary=[]
    lbd = tools.eigen(P=100000,para=para)
    for G in Gs:
        all_boundary.append(FF([G.real,G.imag],para=para,lbd=lbd)[0])
    index = np.argmax(np.array(all_boundary))
    print("boundary{},G{}".format(all_boundary[index],Gs[index]))
    return all_boundary[index],Gs[index]
def chaos_boundary_finder(alpha,c,d,x0=-1,x1=10,step=0.1,gamma0=0.1,gamma1=2):
    def func(gamma):
        para = [alpha,c,d,gamma]
        xx = spectrum_right_boundary_finder(x0,x1,step,para=para)[0]-1
        print(xx)
        return xx
    return optimize.bisect(func,gamma0,gamma1,xtol=1e-3)