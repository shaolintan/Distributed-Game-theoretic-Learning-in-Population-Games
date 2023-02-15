# -*- coding: utf-8 -*-


import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import PDToolbox as PD

P = 1
n = 10
time = 100
p = 0.1
x0 = np.random.rand(1,10)
x0 = x0/np.sum(x0)
L = nx.Graph()
L.add_nodes_from(list(range(1,11)))
G = PD.definition(L,n=n,x0=x0,dynamics=['smith'],time=time,f=PD.fitness4)

T_total = np.array(())
X_total = np.array(())
for i in range(time):
    G.L.remove_edges_from(list(G.L.edges()))
    adj_mat = np.zeros((n,n))
    for j in range(n):
        for k in range(j+1, n):
            if random.random() < p:
                adj_mat[j, k] = 1
                adj_mat[k, j] = 1
                G.L.add_edge(j+1,k+1)
    G.time=1
    G.x0=x0
    G.run()
    T = G.T+i
    X = G.X
    if i == 0:
        T_total = T.copy()
        X_total = X.copy()
    else:
        T_total = np.concatenate([T_total, T])
        X_total = np.r_[X_total, X]
    x0=X[-1,:]

plt.figure()
for s in range(n):
    plt.plot(T_total,X_total[:,s])
plt.xlim(0,time)