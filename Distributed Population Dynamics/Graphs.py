# -*- coding: utf-8 -*-


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def graph_simplex(G):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(G.X[:,0], G.X[:,1], G.X[:,2])
    k1=np.array((0,1,0,0))
    k2=np.array((1,0,0,1))
    k3=np.array((0,0,1,0))
    ax.plot(k1,k2,k3,'k')
    plt.xlim(0,1)
    plt.ylim(0,1)
    ax.view_init(elev=25, azim=45)
    plt.title('simplex')


def graph_evolution(G):
    plt.figure()
    n = np.max(G.S)
    for p in range(G.P):
        plt.subplot(G.P,1,p+1)
        for s in range(G.S[p][0]):
            labels = str(s+1)+'-th strategy'
            plt.plot(G.T,G.X[:,p*n+s], label = labels)
        plt.ylim(0,1)
        plt.xlim(0,G.T[-1])
        plt.title('Evolution of the '+str(p+1)+'-th Population')
        plt.xlabel('time')
        plt.legend()
        plt.show()

def graph_multi_pop(G):
    plt.figure()
    plt.plot(G.X[:, 0], G.X[:, G.S[0][0]+1])
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.title('simplex')

def graph_fitness(G):
    plt.figure()
    n = np.max(G.S)
    len_T = np.size(G.T)
    utility = np.zeros((len_T, G.P*n))
    utility_mean = np.zeros((len_T, G.P))
    State = np.zeros((len_T, n))
    for t in range(len_T):
        x = G.state(G, G.T[t])
        State[t, :] = x[0, :]
        F = np.zeros((G.P, n))
        if G.pop_wise is False:
            F[:,:] = G.f(x,0,G)
        else:
            for p in range(G.P):
                F[p, :] = G.f(x,p+1,G)
        for p in range(G.P):
            k = p*n
            for s in range(G.S[p][0]):
                utility[t, k + s] = F[p, s]
            utility_mean[t, p] = F[p, :].dot(x[p, :])/G.m[p][0]
    
    for p in range(G.P):
        k = p*n
        plt.subplot(G.P,1,p+1)
        for s in range(G.S[p][0]):
            labels = str(s+1)+'-th strategy'
            plt.plot(G.T,utility[:,k+s], label = labels)
        plt.plot(G.T, utility_mean[:, p], 'k-.', label = 'Weighted Avg. Population')
        plt.xlim(0,G.time)
        plt.title('Fitness of the '+str(p+1)+'-th Population')
        plt.xlabel('time')
        plt.legend()
        plt.show()

