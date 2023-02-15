from scipy.integrate import odeint
import networkx as nx
import scipy.io as sio
import numpy as np


def graphical_rep_dyn(y,t,A,a,b,c,d):
    N=size(A,1)
    deg=sum(A,1)
    return (diag((a-b-c+d)*A*y)+diag((b-d)*deg))*(eye(N)-diag(y))*y

g=nx.random_geometric_graph(100,0.15)
A=nx.adjacency_matrix(g)
tspan=np.range[0,30,0.01]
y0=np.random.rand(1,N)
track=odeint(graphical_rep_dyn,(A,1,0,0,1),tspan,args=y0)
