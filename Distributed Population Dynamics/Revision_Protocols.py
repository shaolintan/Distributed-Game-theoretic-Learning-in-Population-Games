# -*- coding: utf-8 -*-


import random
from math import ceil
import numpy as np

def state(s,S, N):
    z = np.zeros(S[0][0])
    for i in range(S[0][0]):
        z[i] = np.sum(s==(i+1))/N
    return z

def proportional_imitation(F, z, s, i, N, R, eta, S, m):
    j = np.random.randint(0,N)
    pi_i = F[s[i]-1]
    pi_j = F[s[j]-1]
    
    rho_ij = max(pi_i-pi_j,0)
    change = ceil(random.random()-1+rho_ij/R)
    if change == 1:
        s_i = s[j]
    else:
        s_i = s[i]
    return s_i

def logit_choice(F, z, s, i, N, R, eta, S, m):
    j = np.random.randint(0,S[0][0])
    F_ = np.exp(F[:S[0][0]]/eta)
    F_mean = np.sum(F_)
    
    rho_ij =  F_[j] / F_mean
    change = ceil(random.random()-1+rho_ij/R)
    if change == 1:
        s_i = np.array((j+1,))
    else:
        s_i = s[i]
    return s_i
    
def comparison2average(F, z, s, i, N, R, eta, S, m):
    j = np.random.randint(0,S[0][0])
    pi_j = F[j]
    rho_ij = max(pi_j-F.dot(z)/m[0][0], 0);
    change = ceil(random.random()-1+rho_ij/R)
    if change == 1:
        s_i = np.array((j+1,))
    else:
        s_i = s[i]
    return s_i
    
def pairwise_comparison(F, z, s, i, N, R, eta, S, m):
    j = np.random.randint(0,S[0][0])
    pi_i = F[s[i]-1]
    pi_j = F[j]
    
    rho_ij = max(pi_i-pi_j,0)
    change = ceil(random.random()-1+rho_ij/R)
    if change == 1:
        s_i = np.array((j+1,))
    else:
        s_i = s[i]
    return s_i

