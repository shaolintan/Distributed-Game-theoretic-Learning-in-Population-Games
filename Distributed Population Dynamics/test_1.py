# -*- coding: utf-8 -*-


import PDToolbox as PD
import numpy as np

# ode: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA', 'odeint'
#       more information,
#       https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
#       https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
# (1)odeint没有微分方程求解算法选择,ode选择'odeint'即可调用odeint函数
# (2)solve_ivp可选多种微分方程求解算法，但可能需要更新scipy版本，ode选择'RK45', 
#    'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'即可调用solve_ivp函数

# 2020/8/10修改:
# (1)将dynamics矩阵操作改为图操作
# (2)dynamics neighbour中加i节点自身：rd, smith, projection;
#    dynamics neighbour不加i节点自身：logit, bnn, maynard_rd, smith_b
# (3)修改definition，将networkx定义的图L作为defination的一个参数，L默认为完全图
# (4)修改definition，在class内定义run等函数，调用时不会出现自己调用自己的写法，例如：
#    执行run函数，之前写法G.run(G) 现在改为 G.run()
# (5)Dynamics中删除dis_smith，用simth函数即可完成分布式smith

#run_game
x0 = np.array([[0.2,0.7,0.1]])
G = PD.definition(n=3,x0 = x0,dynamics=['rd'],time=60,gamma=[0.25,0.75])
G.run()
G.graph()
G.graph_evolution()
G.graph_fitness()


#run_game_finite_population
# =============================================================================
# x0 = np.array([[0.2,0.7,0.1]])
# G = PD.definition(n=3,x0 = x0,revision_protocol='proportional_imitation', time=10000,N=200)
# G.run_finite()
# G.graph()
# G.graph_evolution()
# =============================================================================


#Multi-population Games
# =============================================================================
# x0 = np.array([[0.2,0.8],[0.3,0.7]])
# G = PD.definition(n=2,x0 = x0.T,P=2,dynamics=['rd'],time=60,f=PD.fitness2)
# G.run()
# G.graph()
# G.graph_evolution()
# G.graph_fitness()
# =============================================================================
