# -*- coding: utf-8 -*-


import numpy as np
from math import exp

def rd(t, z, G):
    n = np.max(G.S)
    F = np.zeros((G.P, n))
    x_dot_v = np.zeros(G.P* n)
    x_n = z.copy()
    x_n.resize(G.P,n)
    x = np.zeros((G.P, n))
    for p in range(G.P):
        x[p,:] = x_n[p,:]*G.m[p]
    if G.pop_wise is False:
        F = G.f(x,0,G)
    else:
        for p in range(G.P):
            F[p,:] = G.f(x,p+1,G)
    for p in range(G.P):
        for i in range(n):
            N_i = [j-1 for j in list(G.L[i+1])]
            for j in N_i:
                x_dot_v[p*n+i] = x_dot_v[p*n+i]+x_n[p][j]*(F[p][i]-F[p][j])
            x_dot_v[p*n+i] = x_dot_v[p*n+i]*x_n[p][i]
    
    dz = x_dot_v.copy()
    if G.stop_c is True:
        G.norm_dx = np.linalg.norm(dz)
    return dz

def smith(t, z, G):
    n = np.max(G.S)
    F = np.zeros((G.P, n))
    x_dot_v = np.zeros(G.P* n)
    x_n = z.copy()
    x_n.resize(G.P,n)
    x = np.zeros((G.P, n))
    for p in range(G.P):
        x[p,:] = x_n[p,:]*G.m[p][0]
    if G.pop_wise is False:
        F = G.f(x,0,G)
    else:
        for p in range(G.P):
            F[p,:] = G.f(x,p+1,G)
    for p in range(G.P):
        for i in range(n):
            N_i = [j-1 for j in list(G.L[i+1])]
            for j in N_i:
                temp = x[p][j]*max(0,F[p][i]-F[p][j])-x[p][i]*max(0,F[p][j]-F[p][i])
                x_dot_v[p*n+i] = x_dot_v[p*n+i]+temp
    
    dz = x_dot_v.copy()
    if G.stop_c is True:
        G.norm_dx = np.linalg.norm(dz)
    return dz

def logit(t, z, G):
    eta = G.eta
    n = np.max(G.S)
    x_n = z.copy()
    x_n.resize(G.P,n)
    x = np.zeros((G.P, n))
    for p in range(G.P):
        x[p,:] = x_n[p,:]*G.m[p]
    F = np.zeros((G.P, n))
    x_dot_v = np.zeros(G.P* n)
    if G.pop_wise is False:
        F = G.f(x,0,G)
    else:
        for p in range(G.P):
            F[p,:] = G.f(x,p+1,G)
    
    for p in range(G.P):
        for i in range(n):
            N_i = [j-1 for j in list(G.L[i+1])]+[i]
            sum_pho = 0
            for j in N_i:
                sum_pho = sum_pho+exp(F[p][j]/eta)
                temp = x_n[p][j]*exp(F[p][i]/eta)-x_n[p][i]*exp(F[p][j]/eta)
                x_dot_v[p*n+i] = x_dot_v[p*n+i]+temp
            x_dot_v[p*n+i] = x_dot_v[p*n+i]/sum_pho
    
    dz = x_dot_v.copy()
    if G.stop_c is True:
        G.norm_dx = np.linalg.norm(dz)
    return dz

def projection(t, z, G):
    n = np.max(G.S)
    F = np.zeros((G.P, n))
    x_dot_v = np.zeros(G.P* n)
    x_n = z.copy()
    x_n.resize(G.P,n)
    x = np.zeros((G.P, n))
    for p in range(G.P):
        x[p,:] = x_n[p,:]*G.m[p][0]
    if G.pop_wise is False:
        F = G.f(x,0,G)
    else:
        for p in range(G.P):
            F[p,:] = G.f(x,p+1,G)
    for p in range(G.P):
        for i in range(n):
            N_i = [j-1 for j in list(G.L[i+1])]
            for j in N_i:
                temp = F[p][i]-F[p][j]
                x_dot_v[p*n+i] = x_dot_v[p*n+i]+temp
    
    dz = x_dot_v.copy()
    if G.stop_c is True:
        G.norm_dx = np.linalg.norm(dz)
    return dz

def bnn(t, z, G):
    n = np.max(G.S)
    x_n = z.copy()
    x_n.resize(G.P,n)
    x = np.zeros((G.P, n))
    for p in range(G.P):
        x[p,:] = x_n[p,:]*G.m[p]
    F = np.zeros((G.P, n))
    x_dot_v = np.zeros(G.P* n)
    if G.pop_wise is False:
        F = G.f(x,0,G)
    else:
        for p in range(G.P):
            F[p,:] = G.f(x,p+1,G)
    for p in range(G.P):
        for i in range(n):
            x_dot_v[p*n+i] = max(F[p][i], 0)
            N_i = [j-1 for j in list(G.L[i+1])]+[i]
            F_mean = F[p,N_i].dot(x_n[p,N_i]).T
            x_dot_v[p*n+i] = max(F[p][i]-F_mean,0)
            for j in N_i:
                x_dot_v[p*n+i] = x_dot_v[p*n+i]-x[p][i]*max(F[p][j]-F_mean,0)
    
    dz = x_dot_v.copy()
    if G.stop_c is True:
        G.norm_dx = np.linalg.norm(dz)
    return dz

def maynard_rd(t, z, G):
    n = np.max(G.S)
    F = np.zeros((G.P, n))
    x_dot_v = np.zeros(G.P* n)
    x_n = z.copy()
    x_n.resize(G.P,n)
    x = np.zeros((G.P, n))
    for p in range(G.P):
        x[p,:] = x_n[p,:]*G.m[p]
    if G.pop_wise is False:
        F = G.f(x,0,G)
    else:
        for p in range(G.P):
            F[p,:] = G.f(x,p+1,G)
    for p in range(G.P):
        for i in range(n):
            N_i = [j-1 for j in list(G.L[i+1])]#+[i]
            F_mean = 0
            for j in N_i:
                F_mean = F_mean+F[p][j]*x_n[p][j]
                x_dot_v[p*n+i] = x_dot_v[p*n+i]+x_n[p][j]*(F[p][i]-F[p][j])
            x_dot_v[p*n+i] = x_dot_v[p*n+i]*x_n[p][i]/F_mean
    
    dz = x_dot_v.copy()
    if G.stop_c is True:
        G.norm_dx = np.linalg.norm(dz)
    return dz

def smith_b(t, z, G):
    n = np.max(G.S)
    x_n = z.copy()
    x_n.resize(G.P,n)
    x = np.zeros((G.P, n))
    F = np.zeros((G.P, n))
    F_excess_a = np.zeros(n)
    F_excess_b = np.zeros(n)
    x_dot_v = np.zeros(G.P* n)
    for p in range(G.P):
        x[p,:] = x_n[p,:]*G.m[p]
    if G.pop_wise is False:
        F = G.f(x,0,G)
    else:
        for p in range(G.P):
            F[p,:] = G.f(x,p+1,G)
    for p in range(G.P):
        for i in range(n):
            N_i = [j-1 for j in list(G.L[i+1])]+[i]
            F_mean = 0
            S = []
            S2 = []
            for j in N_i:
                F_mean = F_mean+F[p][j]*x_n[p][j]
                if F[p][j]<F[p][i]:
                    S = S+[j]
                elif F[p][j]>F[p][i]:
                    S2 = S2+[j]
            F_excess_a = 0
            F_excess_b = np.sum(F[p][S2])-F[p][i]*(len(N_i)-len(S)-1)
            for j in S:
                F_excess_a = F_excess_a+F[p][i]*x_n[p][j]-F[p][j]*x_n[p][j]
            x_dot_v[p*n+i] = F_excess_a-F_excess_b*x_n[p][i]
    
    dz = x_dot_v.copy()
    if G.stop_c is True:
        G.norm_dx = np.linalg.norm(dz)
    return dz


def combined_dynamics(t, z, G):
    n = np.max(G.S)
    dz = np.zeros(G.P* n)
    dz_i = np.zeros(G.P* n)
    for i in range(len(G.dynamics)):
    	if G.gamma[i] != 0:
            if G.dynamics[i] == 'rd':
                dz_i = rd(t, z, G)
            elif G.dynamics[i] == 'maynard_rd':
                dz_i = maynard_rd(t, z, G)
            elif G.dynamics[i] == 'bnn':
                dz_i = bnn(t, z, G)
            elif G.dynamics[i] == 'smith':
                dz_i = smith(t, z, G)
            elif G.dynamics[i] == 'smith_b':
                dz_i = smith_b(t, z, G)
            elif G.dynamics[i] == 'logit':
                dz_i = logit(t, z, G)
            dz = dz + G.gamma[i] * dz_i
    return dz

def stopevent(t, z, G):
    return G.norm_dx - G.c_error

# =============================================================================
# #以下是基于矩阵操作的dynamics
# 
# def rd(t, z, G):
#     n = np.max(G.S)
#     F = np.zeros((G.P, n))
#     F_mean = np.zeros((G.P, 1))
#     x_dot_v = np.zeros(G.P* n)
#     x_n = z.copy()
#     x_n.resize(G.P,n)
#     x = np.zeros((G.P, n))
#     for p in range(G.P):
#         x[p,:] = x_n[p,:]*G.m[p]
#     if G.pop_wise is False:
#         F = G.f(x,0,G)
#     else:
#         for p in range(G.P):
#             F[p,:] = G.f(x,p+1,G)
#     for p in range(G.P):
#         F_mean[p] = F[p, :].dot(x_n[p, :].T)
#         F_excess = F[p, :] - np.ones((1,n))*F_mean[p][0]
#         x_dot_v[p*n:(p+1)*n] = np.multiply(F_excess, x_n[p, :])
#     
#     dz = x_dot_v.copy()
#     if G.stop_c is True:
#         G.norm_dx = np.linalg.norm(dz)
#     return dz
# 
# def bnn(t, z, G):
#     n = np.max(G.S)
#     x_n = z.copy()
#     x_n.resize(G.P,n)
#     x = np.zeros((G.P, n))
#     for p in range(G.P):
#         x[p,:] = x_n[p,:]*G.m[p]
#     F_mean = np.zeros((G.P, 1))
#     F = np.zeros((G.P, n))
#     F_excess = np.zeros((G.P, n))
#     F_gamma = np.zeros((G.P, 1))
#     x_dot_v = np.zeros(G.P* n)
#     if G.pop_wise is False:
#         F = G.f(x,0,G)
#     else:
#         for p in range(G.P):
#             F[p,:] = G.f(x,p+1,G)
#     for p in range(G.P):
#         F_mean[p] = F[p, :].dot(x_n[p, :].T)
#         temp = F[p, :] - F_mean[p]
#         F_excess[p,:] = 1 * (temp > 0) * temp
#         F_gamma[p] = F_excess[p, :].dot(np.ones((n, 1)))
#         
#         temp = np.array(F_excess[p, :]-F_gamma[p]*x_n[p, :])
#         x_dot_v[p*n:(p+1)*n] = temp
#     
#     dz = x_dot_v.copy()
#     if G.stop_c is True:
#         G.norm_dx = np.linalg.norm(dz)
#     return dz
# 
# def maynard_rd(t, z, G):
#     n = np.max(G.S)
#     F = np.zeros((G.P, n))
#     F_mean = np.zeros((G.P, 1))
#     x_dot_v = np.zeros(G.P* n)
#     x_n = z.copy()
#     x_n.resize(G.P,n)
#     x = np.zeros((G.P, n))
#     for p in range(G.P):
#         x[p,:] = x_n[p,:]*G.m[p]
#     if G.pop_wise is False:
#         F = G.f(x,0,G)
#     else:
#         for p in range(G.P):
#             F[p,:] = G.f(x,p+1,G)
#     for p in range(G.P):
#         F_mean[p] = F[p, :].dot(x_n[p, :].T)
#         F_excess = F[p, :] - np.ones((1,n))*F_mean[p][0]
#         x_dot_v[p*n:(p+1)*n] = np.multiply(F_excess, x_n[p, :])/F_mean[p]
#     
#     dz = x_dot_v.copy()
#     if G.stop_c is True:
#         G.norm_dx = np.linalg.norm(dz)
#     return dz
# 
# def smith(t, z, G):
#     n = np.max(G.S)
#     F = np.zeros((G.P, n))
#     F_sum = np.zeros((G.P,n))
#     F_mean = np.zeros((G.P, n))
#     x_dot_v = np.zeros(G.P* n)
#     x_n = z.copy()
#     x_n.resize(G.P,n)
#     x = np.zeros((G.P, n))
#     for p in range(G.P):
#         x[p,:] = x_n[p,:]*G.m[p]
#     if G.pop_wise is False:
#         F = G.f(x,0,G)
#     else:
#         for p in range(G.P):
#             F[p,:] = G.f(x,p+1,G)
#     for p in range(G.P):
#         A = np.ones((n,1)).dot(np.array([F[p,:]]))
#         temp = A-A.T
#         M = 1 * (temp > 0) * temp
#         F_sum[p, :] = M.dot(np.ones((n,1))).T
#         F_mean[p, :] = x_n[p,:].dot(M)
#         x_dot_v[p*n:(p+1)*n] = F_mean[p,:] - np.multiply(x_n[p,:],F_sum[p,:])
#     
#     dz = x_dot_v.copy()
#     if G.stop_c is True:
#         G.norm_dx = np.linalg.norm(dz)
#     return dz
# 
# def smith_b(t, z, G):
#     n = np.max(G.S)
#     x_n = z.copy()
#     x_n.resize(G.P,n)
#     x = np.zeros((G.P, n))
#     F = np.zeros((G.P, n))
#     F_excess_a = np.zeros(n)
#     F_excess_b = np.zeros(n)
#     x_dot_v = np.zeros(G.P* n)
#     for p in range(G.P):
#         x[p,:] = x_n[p,:]*G.m[p]
#     if G.pop_wise is False:
#         F = G.f(x,0,G)
#     else:
#         for p in range(G.P):
#             F[p,:] = G.f(x,p+1,G)
#     for p in range(G.P):
#         B = np.argsort(F[p, :G.S[p][0]])
#         A = F[p, :G.S[p][0]][B]
#         A_sum = A[:n].dot(np.ones((n,1)))
#         A_avg = 0
#         x_ordered = x_n[p, B]
#         x_cum = 0
#         for i in range(G.S[p][0]):
#             k = B[i]
#             A_sum = A_sum-A[i]
#                
#             F_excess_a[k] = A[i]*x_cum - A_avg
#             F_excess_b[k] = A_sum - A[i]*(n-i-1)
#             
#             A_avg = A_avg + A[i]*x_ordered[i]
#             x_cum = x_cum + x_ordered[i]
#         print(F_excess_b)
#         x_dot_v[p*n:(p+1)*n] = F_excess_a - np.multiply(F_excess_b,x_n[p, :])
#     
#     dz = x_dot_v.copy()
#     if G.stop_c is True:
#         G.norm_dx = np.linalg.norm(dz)
#     return dz
# 
# def logit(t, z, G):
#     eta = G.eta
#     n = np.max(G.S)
#     x_n = z.copy()
#     x_n.resize(G.P,n)
#     x = np.zeros((G.P, n))
#     for p in range(G.P):
#         x[p,:] = x_n[p,:]*G.m[p]
#     F = np.zeros((G.P, n))
#     F_ = np.zeros((G.P, n))
#     F_mean = np.zeros((G.P, 1))
#     x_dot_v = np.zeros(G.P* n)
#     if G.pop_wise is False:
#         F = G.f(x,0,G)
#     else:
#         for p in range(G.P):
#             F[p,:] = G.f(x,p+1,G)
#     for p in range(G.P):
#         F_[p, :] = np.exp(F[p, :]/eta)
#         F_mean[p] = F_[p,:].dot(np.ones((n,1)))
#         x_dot_v[p*n:(p+1)*n] = F_[p, :]/F_mean[p] - x_n[p, :]
#     
#     dz = x_dot_v.copy()
#     if G.stop_c is True:
#         G.norm_dx = np.linalg.norm(dz)
#     return dz
# 
# import networkx as nx
# def dis_smith(t, z, G):
#     A_mat = nx.adjacency_matrix(G.L).todense().A
#     n = np.max(G.S)
#     F = np.zeros((G.P, n))
#     F_sum = np.zeros((G.P,n))
#     F_mean = np.zeros((G.P, n))
#     x_dot_v = np.zeros(G.P* n)
#     x_n = z.copy()
#     x_n.resize(G.P,n)
#     x = np.zeros((G.P, n))
#     for p in range(G.P):
#         x[p,:] = x_n[p,:]*G.m[p]
#     if G.pop_wise is False:
#         F = G.f(x,0,G)
#     else:
#         for p in range(G.P):
#             F[p,:] = G.f(x,p+1,G)
#     for p in range(G.P):
#         A = np.ones((n,1)).dot(np.array([F[p,:]]))
#         temp = A-A.T
#         M = 1 * (temp > 0) * temp
#         M = A_mat*M
#         F_sum[p, :] = M.dot(np.ones((n,1))).T
#         F_mean[p, :] = x_n[p,:].dot(M)
#         x_dot_v[p*n:(p+1)*n] = F_mean[p,:] - np.multiply(x_n[p,:],F_sum[p,:])
#     
#     dz = x_dot_v.copy()
#     if G.stop_c is True:
#         G.norm_dx = np.linalg.norm(dz)
#     return dz
# =============================================================================
