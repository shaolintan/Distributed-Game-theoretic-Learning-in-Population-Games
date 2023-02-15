import networkx as nx
import scipy.io as sio


g=nx.read_weighted_edgelist('macrophage_symm.txt')
A=nx.adjacency_matrix(g)
sio.savemat('A_macrophage.mat',{'A_macrophage':A})


