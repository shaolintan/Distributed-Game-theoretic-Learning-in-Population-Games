# -*- coding: utf-8 -*-


import networkx as nx
from scipy.integrate import odeint, solve_ivp
import numpy as np
import sys
from math import floor
import time as Time
import Dynamics
import Revision_Protocols
import Graphs


def fitness1(x, p, G):
    A=np.array([[0,-1,1],[1,0,-1],[-1,1,0]])
    A = A+2
    f_i = A.dot(x.T)
    return f_i.T

def fitness2(x, p, G):
    A = np.zeros((2,2,2))
    A[:,:,0] = np.array([[2,1],[1,2]])
    A[:,:,1] = np.array([[1,2],[2,1]])
    Ai = np.squeeze(A[:,:,p-1])
    p_ = p + (-1)**(p+1)-1
    f_i = Ai.dot(x[p_, :])
    return f_i

def fitness4(x,p,G):
    b = np.ones((1,10))
    b[0][0] = 2
    b[0][1] = 2
    f_i = -b-x
    return f_i

def fitness5(x, p, G):#p从1开始
    A = np.array([[-17,1,1,1,1],[1,-11,1,1,1],[1,1,-8,1,1],[1,1,1,-6.2,1],
                 [1,1,1,1,-5]])
    neighbors = [i for i in G.L[p]]
    f_i = A.dot(x[p-1,:])-G.L.degree(p)*x[p-1,:]
    for i in range(len(neighbors)):
        f_i = f_i+x[neighbors[i]-1,:]
    return f_i

def run_game(G):
    time_vec = np.linspace(G.step, G.time+G.step, int(G.time/G.step)+1)
    if G.verb is True:
        string = ''
        for i in range(len(G.dynamics)):
            string = string+G.dynamics[i]
            if i < len(G.dynamics)-1:
                string = string+', '
        print('Running '+string+' dynamics')
    start = Time.process_time()
    x0 = G.x0.copy()
    x0.resize(G.P*np.max(G.S),)
    t_span = (G.step, G.time+G.step)
    if len(G.dynamics) == 1:
        func = eval('Dynamics.'+G.dynamics[0])
        Dynamics.stopevent.terminal = True
        if G.ode=='odeint':
            dx = odeint(func,x0,time_vec,args=(G,),tfirst=True)
        else:
            dx = solve_ivp(func, t_span=t_span, t_eval=time_vec, y0=x0, 
                           args=(G,), method=G.ode, rtol=G.RelTol, 
                           atol=G.AbsTol, events=Dynamics.stopevent)
    else:
        if G.ode=='odeint':
            dx = odeint(Dynamics.combined_dynamics, x0, time_vec, args=(G,),
                        tfirst=True)
        else:
            dx = solve_ivp(Dynamics.combined_dynamics, t_span=t_span,
                           t_eval=time_vec, y0=x0, args=(G,), method=G.ode,
                           rtol=G.RelTol, atol=G.AbsTol, 
                           events=Dynamics.stopevent)
    end = Time.process_time()
    print('Time has passed '+str(end-start)+' seconds.')
    if G.ode=='odeint':
        G.T = time_vec
        G.X = dx
    else:
        G.T = dx.t
        G.X = dx.y.T
    for i in range(G.S[0][0]):
        G.L.nodes[i+1]['X'] = G.X[:,i]

def run_game_finite_population(G):
    if G.verb is True:
        print('Running the '+G.revision_protocol+' revision protocol')
        start = Time.process_time()
    func = eval('Revision_Protocols.'+G.revision_protocol)
    s = np.zeros((G.N, G.P))
    h = 0
    for i in range(np.max(G.S[0])):
        p = int(G.N*G.x0[0, i])
        if p+h <= G.N and p!=0:
            s[h: h+p,0] = i+1
            h = h+p
    if h != G.N:
        s[h: G.N,0] = np.random.randint(1,G.S[0][0]+1,size=G.N-h)
    t_max = int(G.time)
    T = np.array(range(t_max))+1
    X = np.zeros((t_max, G.S[0][0]))
    alarm = np.random.poisson(lam=G.N*G.P, size=t_max)
    s = s.astype('int32')
    for t in range(t_max):
        x = Revision_Protocols.state(s,G.S,G.N)
        F = G.f(x,0,G)
        update_agents = np.random.randint(0,G.N,size=alarm[t])
        for k in range(alarm[t]):
            i = update_agents[k]
            s_update = s.copy()
            s_update[i] = func(F, x, s, i, G.N, G.R, G.eta,
                    G.S, G.m)
        s = s_update.copy()
        X[t,:] = x
    if G.verb is True:
        end = Time.process_time()
    print('Time has passed '+str(end-start)+' seconds.')
    G.X = X
    G.T = T

def state(G, T):
    n = np.max(G.S)
    if T < 0:
        t = np.size(G.T)
    elif T >= G.T[-1]:
        t = np.size(G.T)-1
    else:
        t = floor(T/G.step)-1
    x_n = G.X[t, :].copy()
    x_n.resize(G.P, n)
    x = np.zeros((G.P, n))
    for p in range(G.P):
        x[p, :] = x_n[p, :]*G.m[p][0]
    return x

class definition():
    def __init__(self,L=nx.Graph(),P=1,n=None,S=None,x0=None,m=None,
                 dynamics=['rd'],revision_protocol='proportional_imitation',
                 gamma=[],time=30,step=0.01,f=fitness1,pop_wise = True,R=1,
                 N=None,verb=True,eta=0.02,ode='odeint',tol=None,RelTol=1e-4,
                 AbsTol=1e-4,stop_c=False,c_error=1e-5,norm_dx=1):
        self.L = L
        self.P = P
        self.n = n
        self.S = S
        self.x0 = x0
        self.m = m
        self.time = time
        self.step = step
        self.pop_wise = pop_wise
        self.N = N
        self.f = f
        self.eta = eta
        self.dynamics = dynamics
        self.revision_protocol = revision_protocol
        self.gamma = gamma
        self.R = R
        self.verb = verb
        self.ode = ode
        self.tol = tol
        self.RelTol = RelTol
        self.AbsTol = AbsTol
        self.stop_c = stop_c
        self.c_error = c_error
        self.norm_dx = norm_dx
        self.state = state
        if self.P<1:
            print('Error: Invalid value of G.P.')
            sys.exit()
        if self.n!=None:
            if floor(self.n)>1:
                self.S = np.ones((self.P,1))*self.n
            else:
                print('Error: Invalid value of G.n.')
                sys.exit()
        else:
            if self.S is None:
                print('Error: Number of strategies per population must be defined.')
                sys.exit()
            elif np.size(self.S, 0) < self.P:
                print('Error: Number of strategies not defined for some populations.')
                sys.exit()
        self.S = self.S.astype(int)
        
        if self.L.number_of_nodes() == 0:
            self.L.add_nodes_from(list(range(1,np.max(self.S)+1)))
            for i in range(np.max(self.S)-1):
                for j in range(i+1,np.max(self.S)):
                    self.L.add_edge(i+1, j+1)
        
        if x0 is None:
            self.x0 = np.zeros((self.P,np.max(self.S)))
            for i in range(self.P):
                x = np.random.rand(1,self.S[i][0])
                self.x0[i,:self.S[i][0]] = x/np.sum(x)
        else:
            n = np.size(self.x0, 0)
            m = np.size(self.x0, 1)
            if not (n==self.P and m==np.max(self.S)):
                if m==self.P and n==np.max(self.S):
                    self.x0 = self.x0.T
                else:
                    print('Error: Invalid initial condition. Size of G.x0 do not match with G.P and G.S.')
                    sys.exit()
        if self.m is None:
            self.m = np.ones((self.P,1))
        elif np.size(self.m, 0)==1:
            m = self.m
            self.m = np.ones(self.P, 1)*m
        elif np.size(self.m,0) < self.P:
            self.m = np.ones((self.P, 1))
            print('Setting by the mass of all populations to 1.')
        
        temp = np.sum(self.x0, axis=1)
        temp.resize(self.P, 1)
        temp = temp-np.ones((self.P, 1))
        for i in range(self.P):
            if np.maximum(temp, -temp)[i][0] >= np.spacing(np.ones((self.P,1)))[i][0]:
                print('Warning: Populations initial state x0 does not match the mass m.')
                break
        if len(dynamics) > 1:
            if len(gamma)==0:
                self.gamma = [1/len(dynamics)]*len(dynamics)
            elif len(dynamics) != len(gamma):
                print('Error: Size of G.gamma do not match the size do G.dynamics')
                sys.exit()
        if self.tol is not None:
            self.RelTol = self.tol
            self.AbsTol = self.tol
        if self.N is None:
            self.R=100
    
    def run(self):
        run_game(self)
    
    def run_finite(self):
        run_game_finite_population(self)
    
    def graph(self):
        Graphs.graph_simplex(self)
    
    def graph_evolution(self):
        Graphs.graph_evolution(self)
    
    def graph_fitness(self):
        Graphs.graph_fitness(self)
    
