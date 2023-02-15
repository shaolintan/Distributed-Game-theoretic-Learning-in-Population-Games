# -*- coding: utf-8 -*-


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import PDToolbox as PD

P = 1
n = 10

#Fig2.(d)
time = 20
L = nx.Graph()
L.add_nodes_from(list(range(1,n+1)))
L.add_edges_from([(1,5),(1,6),(1,7),(1,3),(1,4),(2,6),(2,7),(2,8),(2,9),(2,10),
                  (3,4),(3,5),(4,5),(6,7),(8,9),(8,10),(9,10)])
x0 = np.random.rand(1,n)
x0 = x0/np.sum(x0)
G = PD.definition(L=L,P=P,n=n,x0=x0,dynamics=['smith_b'],time=time,f=PD.fitness4)
G.run()
plt.figure()
for s in range(G.S[0][0]):
    plt.plot(G.T,G.X[:,s],label = str(s+1)+'-th strategy')
    plt.xlabel('time')
plt.xlim(0,time)
plt.legend(loc='upper right')

#Fig2.(e)
time = 100
L = nx.Graph()
L.add_nodes_from(list(range(1,n+1)))
L.add_edges_from([(1,5),(1,6),(1,7),(1,3),(1,4),(2,6),(2,7),(2,8),(2,9),(2,10),
                  (3,4),(3,5),(4,5),(6,7),(8,9),(8,10),(9,10),(3,6)])
x0 = np.random.rand(1,n)
x0 = x0/np.sum(x0)
G = PD.definition(L=L,P=P,n=n,x0=x0,dynamics=['smith_b'],time=time,f=PD.fitness4)
G.run()
plt.figure()
for s in range(G.S[0][0]):
    plt.plot(G.T,G.X[:,s],label = str(s+1)+'-th strategy')
    plt.xlabel('time')
plt.xlim(0,time)
plt.legend(loc='upper right')

#Fig2.(f)
time = 200
L = nx.Graph()
L.add_nodes_from(list(range(1,n+1)))
L.add_edges_from([(1,5),(1,6),(1,7),(1,3),(1,4),(2,6),(2,7),(2,8),(2,9),(2,10),
                  (3,4),(3,5),(4,5),(6,7),(8,9),(8,10),(9,10),(3,6),(7,9)])
x0 = np.random.rand(1,n)
x0 = x0/np.sum(x0)
G = PD.definition(L=L,P=P,n=n,x0=x0,dynamics=['smith_b'],time=time,f=PD.fitness4)
G.run()
plt.figure()
for s in range(G.S[0][0]):
    plt.plot(G.T,G.X[:,s],label = str(s+1)+'-th strategy')
    plt.xlabel('time')
plt.xlim(0,time)
plt.legend(loc='upper right')
