import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import linalg
from scipy import optimize
import scipy
import os
import tools

class modelpara:
    def __init__(self,N=1000,alpha=0.01,c=0,d=1,gamma=1,if_circulant=1,tau=0.01,phi=np.tanh,diluted=1.0,segment=1,clean_diag=False,gaussian=True):
        self.N = N
        self.alpha=alpha
        self.c=c
        self.d=d
        self.gamma=gamma
        self.tau=tau
        self.phi=phi
        self.diluted=diluted
        self.segment=segment
        self.if_circulant=if_circulant
        self.clean_diag=clean_diag
        self.gaussian=gaussian
    def item(self,ifprint=False,category=None):
        allitem = {}
        for name,value in vars(self).items():
            allitem[name]=value
        return allitem

class rnn:
    def __init__(self,modelpara):
        self.para=modelpara
        self.para_dict = self.para.item(ifprint=False)
        self.X = tools.X(self.para_dict["N"],self.para_dict["alpha"],self.para_dict["c"],self.para_dict["d"],self.para_dict["gamma"],tau=1,if_circulant=self.para_dict["if_circulant"],segment=self.para_dict["segment"])
        self.pattern = tools.random_pattern(self.para_dict["N"],self.para_dict["alpha"],gaussian=self.para_dict["gaussian"])
        self.J = tools.generate_weight(self.X,self.pattern,diluted=self.para_dict["diluted"],clean_diag=self.para_dict["clean_diag"])
        self.ext=lambda t: 0
        
        print(self.para_dict)
    def update_func2(self):
    # $ \frac{dx}{dt} = -x + \sum_{j} J_{ij} /phi(x_j) + I_0 $
        def f(t,r):
            phi = self.para_dict["phi"]
            ext = self.ext
            r_sum = self.J@phi(r)
            dr =  (-r+r_sum+ext(t))/self.para_dict["tau"]
            return dr
        return f
    
    def overlap_pattern(self,norm=True):
        return 1/self.para_dict["N"]*self.pattern@self.state
    def overlap_activation(self):
        phi = self.para_dict["phi"]
        return 1/self.para_dict["N"]*self.pattern@phi(self.state)
    def simulate(self,t,r0,t0=0,dt=1e-1):
        self.func = self.update_func2
        sol = scipy.integrate.solve_ivp(
            self.func(),
            t_span=(t0,t),
            t_eval=np.arange(t0,t,dt),
            y0=r0,
            method="RK23")
        self.t = sol.t
        self.state = sol.y
    def auto_correlation(self):
        def auto_correlation_single(state):
            x0 = state[0]
            all_acor = []
            for x in state:
                all_acor.append(np.average(x*x0))
            return np.array(all_acor)
        return auto_correlation_single(self.para_dict["phi"](self.state.T))
    def reset_weight(self,pattern):
        self.pattern = pattern
        self.J = tools.generate_weight(self.X,self.pattern,diluted=self.para_dict["diluted"],clean_diag=self.para_dict["clean_diag"])
        
def lyapunov_spectrum(model):
    N = model.para_dict["N"]
    Q = np.eye(N)
    T = len(model.state.T)
    lyspectrum = np.zeros(N)
    phip = lambda x: 1-np.tanh(x)**2
    timestep = model.t[1]-model.t[0]
    tau = model.para_dict["tau"]
    gap = 1
    total = 0
    for i,preact in enumerate(model.state.T):
        print("{} start".format(i))
        Ja = timestep/tau*model.J@np.diag(phip(preact))+(1.0- timestep/tau)*np.eye(N)
        Q = Ja@Q
        if i%gap==0:
            total=total+1
            Q,R = np.linalg.qr(Q)
            sign = np.diag(np.sign(np.diag(R)))
            Q = Q@sign
            R = sign@R
            lyspectrum += np.log(np.diag(R))
    return lyspectrum/(total)        